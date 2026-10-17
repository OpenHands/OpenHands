// @vitest-environment node
//
// Drift-detection: documented agent-server version examples must match the
// authoritative `versions.agentServer` pin in `config/defaults.json`.
//
// PR #670 bumped the central pin to 1.23.0 but left stale `1.22.1` examples in
// contributor guidance and several script JSDocs. This test fails when those
// references drift from the central pin so the next bump cannot silently leave
// docs behind.
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { DEFAULT_API_KEY_PATH } from "../../scripts/dev-safe.mjs";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

function read(rel: string): string {
  return readFileSync(path.join(repoRoot, rel), "utf-8");
}

const config = JSON.parse(read("config/defaults.json")) as {
  images: { agentCanvas: string };
  versions: { agentServer: string; agentCanvas: string };
};
const agentServerVersion = config.versions.agentServer;
const dockerImage = `${config.images.agentCanvas}:${config.versions.agentCanvas}`;

describe("docs/example references stay in sync with config/defaults.json", () => {
  it("the local-stack skill documents the current default version", () => {
    const localStackGuide = read(
      ".agents/skills/local-stack-runtime/references/guide.md",
    );
    expect(localStackGuide).toContain(
      `\`OH_AGENT_SERVER_VERSION\` — specific PyPI version (e.g., "${agentServerVersion}")`,
    );
    expect(localStackGuide).toContain(
      `Default: released PyPI version \`${agentServerVersion}\` for agent-server SDK libraries`,
    );
  });

  it("every README Docker image reference uses the pinned tag", () => {
    const imageRefPattern = /ghcr\.io\/openhands\/agent-canvas:[^\s`"]+/g;

    for (const file of ["README.md", "README.windows.md"]) {
      const refs = read(file).match(imageRefPattern) ?? [];
      expect(refs.length).toBeGreaterThan(0);
      for (const ref of refs) {
        expect(ref).toBe(dockerImage);
      }
    }
  });

  it("scripts/dev-safe.mjs JSDoc example matches the current default", () => {
    const devSafe = read("scripts/dev-safe.mjs");
    expect(devSafe).toContain(
      `OH_AGENT_SERVER_VERSION: Specific PyPI version (e.g., "${agentServerVersion}")`,
    );
  });

  it("scripts/check-sdk-version-sync.mjs examples match the current default", () => {
    const src = read("scripts/check-sdk-version-sync.mjs");
    expect(src).toContain(`EXPECTED_SDK_VERSION=${agentServerVersion}`);
    expect(src).toContain(`"version": "${agentServerVersion}"`);
    expect(src).toContain(`"openhands-sdk>=${agentServerVersion},<2.0.0"`);
    expect(src).toContain(`"openhands-tools==${agentServerVersion}"`);
    expect(src).toContain(`"openhands-workspace (>=${agentServerVersion})"`);
    expect(src).toContain(
      `">=${agentServerVersion}", "==${agentServerVersion}", "(>=${agentServerVersion})", "~=${agentServerVersion}"`,
    );
  });
});

// The launchers and the Docker entrypoint persist the auto-generated session
// API key as `api-key.txt`. The contributor guides and the Helm NOTES kept the
// older `session-api-key.txt` name, pointing readers at a file that is never
// written.
describe("documented session API key file matches the launcher", () => {
  it.each([
    ".agents/skills/local-stack-runtime/references/guide.md",
    ".agents/skills/frontend-api-contracts/references/guide.md",
    "helm/agent-canvas/templates/NOTES.txt",
  ])("%s names the persisted key file", (file) => {
    const keyPathFromHome = path
      .relative(homedir(), DEFAULT_API_KEY_PATH)
      .split(path.sep)
      .join("/");
    const refs = read(file).match(/[\w./~-]*api-key\.txt/g) ?? [];

    expect(refs.length).toBeGreaterThan(0);
    for (const ref of refs) {
      expect(ref.endsWith(`/${keyPathFromHome}`), ref).toBe(true);
    }
  });
});
