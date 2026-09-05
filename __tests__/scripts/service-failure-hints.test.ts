// @vitest-environment node
import { afterEach, describe, expect, it } from "vitest";
import { once } from "node:events";
import {
  describeServiceFailure,
  detectToolchainFailure,
  isSignificantLine,
  SERVICE_OUTPUT_BUFFER_LINES,
  SERVICE_SIGNIFICANT_LINE_LIMIT,
} from "../../scripts/service-failure-hints.mjs";
import {
  getServiceFailure,
  setServiceLogListener,
  spawnService,
} from "../../scripts/dev-with-automation.mjs";

// Trimmed from the report in #16300 — a uv build failure followed by the
// cargo/rustc rejection that actually explains it.
const RUSTC_FAILURE = [
  "Building litellm==1.95.0",
  "× Failed to build `litellm==1.95.0`",
  "├─▶ The build backend returned an error",
  "╰─▶ Call to `maturin.build_wheel` failed (exit status: 1)",
  "error: rustc 1.93.1 is not supported by the following packages:",
  "aws-config@1.9.0 requires rustc 1.94.1",
  "aws-smithy-types@1.6.1 requires rustc 1.94.1",
  "Either upgrade rustc or select compatible dependency versions with",
];

describe("detectToolchainFailure", () => {
  it("recognises an out-of-date rustc behind a failed wheel build", () => {
    const hit = detectToolchainFailure(RUSTC_FAILURE);
    expect(hit).not.toBeNull();
    expect(hit?.kind).toBe("rust-toolchain-too-old");
    expect(hit?.found).toBe("1.93.1");
    expect(hit?.required).toBe("1.94.1");
  });

  it("reports the strictest requirement when crates disagree", () => {
    const hit = detectToolchainFailure([
      ...RUSTC_FAILURE,
      "aws-sdk-sts@1.108.0 requires rustc 1.95.0",
    ]);
    expect(hit?.required).toBe("1.95.0");
  });

  it("accepts a single string as well as an array of lines", () => {
    expect(detectToolchainFailure(RUSTC_FAILURE.join("\n"))).not.toBeNull();
  });

  it("stays quiet on a build failure with no rustc complaint", () => {
    expect(
      detectToolchainFailure([
        "× Failed to build `litellm==1.95.0`",
        "╰─▶ Call to `maturin.build_wheel` failed (exit status: 1)",
        "error: linker `cc` not found",
      ]),
    ).toBeNull();
  });

  it("stays quiet on a rustc complaint with no failed build", () => {
    // A warning during an otherwise successful run must not be reported as
    // the reason a service died.
    expect(
      detectToolchainFailure(["rustc 1.93.1 is not supported by something"]),
    ).toBeNull();
  });

  it("stays quiet on empty, missing, or unrelated output", () => {
    expect(detectToolchainFailure([])).toBeNull();
    expect(detectToolchainFailure(undefined)).toBeNull();
    expect(
      detectToolchainFailure(["Uvicorn running on http://127.0.0.1:8001"]),
    ).toBeNull();
  });
});

describe("describeServiceFailure", () => {
  it("names the service and offers both remediations", () => {
    const d = describeServiceFailure("automation", 1, RUSTC_FAILURE);
    expect(d).not.toBeNull();
    expect(d?.service).toBe("automation");
    expect(d?.exitCode).toBe(1);
    expect(d?.lines[0]).toContain("automation could not start");
    expect(d?.lines[0]).toContain("1.93.1");
    const joined = d!.lines.join("\n");
    expect(joined).toContain("rustup update stable");
    expect(joined).toContain("brew upgrade rust");
  });

  it("returns null for unrecognised failures so the generic message stands", () => {
    expect(describeServiceFailure("automation", 1, ["boom"])).toBeNull();
  });
});

describe("SERVICE_OUTPUT_BUFFER_LINES", () => {
  it("keeps enough output for the rustc block to survive truncation", () => {
    // The AWS crate list in #16300 alone is ~20 lines; the buffer has to hold
    // the whole build tail or the matcher never sees the explanation.
    expect(SERVICE_OUTPUT_BUFFER_LINES).toBeGreaterThanOrEqual(50);
  });
});

describe("spawnService integration", () => {
  afterEach(() => {
    setServiceLogListener(null);
  });

  it("explains a toolchain failure when a service dies during start-up", async () => {
    const emitted: string[] = [];
    setServiceLogListener((_name: string, line: string) => {
      emitted.push(line);
    });

    // A stand-in for uvx: print the build failure exactly as reported in
    // #16300, then exit non-zero the way the real automation service does.
    const script = [
      "console.error('Building litellm==1.95.0');",
      "console.error('\u00d7 Failed to build `litellm==1.95.0`');",
      "console.error('\u2570\u2500\u25b6 Call to `maturin.build_wheel` failed (exit status: 1)');",
      "console.error('error: rustc 1.93.1 is not supported by the following packages:');",
      "console.error('aws-config@1.9.0 requires rustc 1.94.1');",
      "process.exit(1);",
    ].join("");

    const proc = spawnService("automation-under-test", process.execPath, [
      "-e",
      script,
    ]);
    await once(proc, "close");
    // `close` guarantees stdio is drained, so no arbitrary wait is needed.

    const joined = emitted.join("\n");
    expect(joined).toContain("could not start");
    expect(joined).toContain("1.93.1");
    expect(joined).toContain("rustup update stable");

    const failure = getServiceFailure("automation-under-test");
    expect(failure).not.toBeNull();
    expect(failure?.kind).toBe("rust-toolchain-too-old");
    expect(failure?.exitCode).toBe(1);
  });

  it("clears a previous failure when the same service is respawned", async () => {
    // A service that failed once and then started cleanly must not keep
    // reporting the old failure to anything reading the registry.
    const failing = [
      "console.error('Call to `maturin.build_wheel` failed (exit status: 1)');",
      "console.error('error: rustc 1.93.1 is not supported by the following packages:');",
      "process.exit(1);",
    ].join("");
    const first = spawnService("respawn-under-test", process.execPath, [
      "-e",
      failing,
    ]);
    await once(first, "close");
    expect(getServiceFailure("respawn-under-test")).not.toBeNull();

    const second = spawnService("respawn-under-test", process.execPath, [
      "-e",
      "process.exit(0);",
    ]);
    await once(second, "close");
    expect(getServiceFailure("respawn-under-test")).toBeNull();
  });

  it("leaves no failure recorded for a service that exits cleanly", async () => {
    const proc = spawnService("clean-service-under-test", process.execPath, [
      "-e",
      "process.exit(0);",
    ]);
    await once(proc, "close");

    expect(getServiceFailure("clean-service-under-test")).toBeNull();
  });
});

describe("isSignificantLine", () => {
  it("keeps the lines a post-mortem needs", () => {
    expect(isSignificantLine("x Failed to build `litellm==1.95.0`")).toBe(true);
    expect(
      isSignificantLine(
        "Call to `maturin.build_wheel` failed (exit status: 1)",
      ),
    ).toBe(true);
    expect(
      isSignificantLine(
        "error: rustc 1.93.1 is not supported by the following",
      ),
    ).toBe(true);
    expect(isSignificantLine("aws-config@1.9.0 requires rustc 1.94.1")).toBe(
      true,
    );
  });

  it("ignores ordinary build chatter", () => {
    expect(isSignificantLine("   Compiling serde v1.0.0")).toBe(false);
    expect(isSignificantLine("Uvicorn running on http://127.0.0.1:8001")).toBe(
      false,
    );
  });

  it("retains evidence separated by more output than the tail holds", () => {
    // uv prints its `Failed to build` summary first, replays the captured
    // build log, and cargo emits a line per crate in between — so the two
    // halves of the evidence can be far more than a tail apart.
    const stream = [
      "x Failed to build `litellm==1.95.0`",
      "Call to `maturin.build_wheel` failed (exit status: 1)",
      ...Array.from(
        { length: SERVICE_OUTPUT_BUFFER_LINES + 50 },
        (_, i) => `   Compiling crate-${i} v1.0.0`,
      ),
      "error: rustc 1.93.1 is not supported by the following packages:",
      "aws-config@1.9.0 requires rustc 1.94.1",
    ];

    const recent: string[] = [];
    const significant: string[] = [];
    for (const line of stream) {
      recent.push(line);
      if (recent.length > SERVICE_OUTPUT_BUFFER_LINES) recent.shift();
      if (isSignificantLine(line)) {
        significant.push(line);
        if (significant.length > SERVICE_SIGNIFICANT_LINE_LIMIT) {
          significant.shift();
        }
      }
    }

    // The bounded tail alone has lost the build-failure half by now.
    expect(detectToolchainFailure(recent)).toBeNull();
    // Retaining the significant lines keeps both halves available.
    const hit = detectToolchainFailure([...significant, ...recent]);
    expect(hit).not.toBeNull();
    expect(hit?.required).toBe("1.94.1");
  });
});
