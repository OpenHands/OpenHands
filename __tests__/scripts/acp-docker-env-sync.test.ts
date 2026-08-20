// @vitest-environment node
//
// Drift-detection for the examples/acp-docker quickstart. Invariants enforced:
// the generated pin must equal versions.agentServer from the single source of
// truth (config/defaults.json); the no-config compose fallback must stay on
// `latest-python`; and the pinned tag must never fall below the Canvas
// compatibility floor (compatibility.minimumAgentServer). Any of those drifting
// fails this test.
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";

import {
  computeAgentServerImage,
  renderEnvLine,
  upsertEnvLine,
} from "../../scripts/gen-acp-docker-env.mjs";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

function read(rel: string): string {
  return readFileSync(path.join(repoRoot, rel), "utf-8");
}

const config = JSON.parse(read("config/defaults.json")) as {
  images: { agentServer: string };
  versions: { agentServer: string };
  compatibility: { minimumAgentServer: string };
};

const pinnedImage = `${config.images.agentServer}:${config.versions.agentServer}-python`;

// Numeric-semver comparison. Throws (rather than silently comparing NaN) if a
// pin carries a non-numeric segment — these defaults.json fields are dotted
// numeric version pins, so a sha or pre-release tag landing here is a config
// error the floor check should surface loudly.
function parseSemver(v: string): number[] {
  if (!/^\d+(\.\d+)*$/.test(v)) {
    throw new Error(`expected a dotted numeric version, got "${v}"`);
  }
  return v.split(".").map(Number);
}

function gte(a: string, b: string): boolean {
  const pa = parseSemver(a);
  const pb = parseSemver(b);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const da = pa[i] ?? 0;
    const db = pb[i] ?? 0;
    if (da !== db) return da > db;
  }
  return true;
}

describe("examples/acp-docker stays in sync with config/defaults.json", () => {
  it("computeAgentServerImage derives the pinned SoT tag", () => {
    expect(computeAgentServerImage(config)).toBe(pinnedImage);
  });

  it("renders the AGENT_SERVER_IMAGE line with the pinned tag", () => {
    expect(renderEnvLine(config)).toBe(`AGENT_SERVER_IMAGE=${pinnedImage}`);
  });

  it("the pinned SoT version satisfies the Canvas compatibility floor", () => {
    expect(
      gte(config.versions.agentServer, config.compatibility.minimumAgentServer),
    ).toBe(true);
  });

  it("the no-config compose fallback uses the SoT registry with the latest-python tag", () => {
    const compose = read("examples/acp-docker/docker-compose.yml");
    const match = compose.match(/AGENT_SERVER_IMAGE:-([^}]+)\}/);
    expect(match?.[1]).toBe(`${config.images.agentServer}:latest-python`);
  });
});

describe("gen-acp-docker-env.mjs is safe to import", () => {
  // The module exports helpers (imported above, and by future consumers), so
  // importing it must not run main(). The entrypoint guard compares
  // import.meta.url against process.argv[1] — but argv[1] is undefined in some
  // ESM contexts (e.g. `node --input-type=module -e "import(...)"`), and an
  // unguarded pathToFileURL(argv[1]) throws ERR_INVALID_ARG_TYPE at import,
  // before any export is reachable. Reproduces that exact context.
  const scriptPath = path.join(repoRoot, "scripts", "gen-acp-docker-env.mjs");

  it("imports without throwing when process.argv[1] is undefined", () => {
    const url = pathToFileURL(scriptPath).href;
    const res = spawnSync(
      process.execPath,
      ["--input-type=module", "-e", `import(${JSON.stringify(url)})`],
      { encoding: "utf-8" },
    );
    expect(res.stderr).not.toMatch(/ERR_INVALID_ARG_TYPE/);
    expect(res.status).toBe(0);
  });
});

describe("upsertEnvLine", () => {
  const line =
    "AGENT_SERVER_IMAGE=ghcr.io/openhands/agent-server:1.28.1-python";

  it("appends the line to an empty file", () => {
    expect(upsertEnvLine("", line)).toBe(`${line}\n`);
  });

  it("replaces an existing assignment in place, preserving other lines", () => {
    const existing =
      "SESSION_API_KEY=abc\n" +
      "AGENT_SERVER_IMAGE=ghcr.io/openhands/agent-server:1.25.0-python\n" +
      "CLAUDE_CODE_OAUTH_TOKEN=zzz\n";
    expect(upsertEnvLine(existing, line)).toBe(
      `SESSION_API_KEY=abc\n${line}\nCLAUDE_CODE_OAUTH_TOKEN=zzz\n`,
    );
  });

  it("is idempotent — re-running yields one assignment, unchanged content", () => {
    const once = upsertEnvLine("", line);
    expect(upsertEnvLine(once, line)).toBe(once);
    expect(once.match(/^AGENT_SERVER_IMAGE=/gm)?.length).toBe(1);
  });

  it("treats a commented template line as documentation and appends the real value", () => {
    // .env.example ships `# AGENT_SERVER_IMAGE=...` as a documented knob; after
    // `cp .env.example .env` the comment stays and the generator adds the value.
    const existing =
      "# AGENT_SERVER_IMAGE=ghcr.io/openhands/agent-server:latest-python\n";
    expect(upsertEnvLine(existing, line)).toBe(
      `${existing.trimEnd()}\n${line}\n`,
    );
  });

  it("throws rather than rewrite every line when given a keyless line", () => {
    expect(() => upsertEnvLine("FOO=bar\n", "novalue")).toThrow();
  });
});
