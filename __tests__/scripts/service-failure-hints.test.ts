// @vitest-environment node
import { afterEach, describe, expect, it } from "vitest";
import { once } from "node:events";
import {
  describeServiceFailure,
  detectToolchainFailure,
  SERVICE_OUTPUT_BUFFER_LINES,
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
    await once(proc, "exit");
    // The exit handler runs on the same tick as the event; let stdio flush.
    await new Promise((r) => setTimeout(r, 50));

    const joined = emitted.join("\n");
    expect(joined).toContain("could not start");
    expect(joined).toContain("1.93.1");
    expect(joined).toContain("rustup update stable");

    const failure = getServiceFailure("automation-under-test");
    expect(failure).not.toBeNull();
    expect(failure?.kind).toBe("rust-toolchain-too-old");
    expect(failure?.exitCode).toBe(1);
  });

  it("leaves no failure recorded for a service that exits cleanly", async () => {
    const proc = spawnService("clean-service-under-test", process.execPath, [
      "-e",
      "process.exit(0);",
    ]);
    await once(proc, "exit");
    await new Promise((r) => setTimeout(r, 50));

    expect(getServiceFailure("clean-service-under-test")).toBeNull();
  });
});
