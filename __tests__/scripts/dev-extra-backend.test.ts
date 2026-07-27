// @vitest-environment node
// These tests load `scripts/dev-safe.mjs` (via dev-extra-backend), which
// constructs file:// URLs relative to its own location via
// `new URL("../tools", import.meta.url)`. jsdom's URL constructor ignores
// file:// base URLs (it falls back to its document base, e.g.
// http://localhost:3000/), breaking that resolution; the Node environment
// has the standard WHATWG URL behavior that honors the file:// base.
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { buildExtraBackendConfig } from "../../scripts/dev-extra-backend.mjs";
import {
  buildSafeDevConfig,
  resetPersistedSessionApiKeyCache,
} from "../../scripts/dev-safe.mjs";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

describe("buildExtraBackendConfig", () => {
  const keyDirs: string[] = [];

  afterEach(() => {
    while (keyDirs.length > 0) {
      const dir = keyDirs.pop();
      if (dir) rmSync(dir, { recursive: true, force: true });
    }
    resetPersistedSessionApiKeyCache();
  });

  function isolatedKeyPath(): string {
    const dir = mkdtempSync(path.join(tmpdir(), "extra-backend-key-"));
    keyDirs.push(dir);
    return path.join(dir, "session-api-key.txt");
  }

  it("defaults to ports 18002/18003 distinct from the bundled instance", () => {
    const env = { OH_SESSION_API_KEY_PATH: isolatedKeyPath() };
    const bundled = buildSafeDevConfig(repoRoot, env);
    const extra = buildExtraBackendConfig(repoRoot, env);

    expect(extra.backendPort).toBe(18002);
    expect(extra.vscodePort).toBe(18003);
    expect(extra.backendBaseUrl).toBe("http://127.0.0.1:18002");
    expect(extra.backendHost).toBe("127.0.0.1:18002");
    expect(extra.backendPort).not.toBe(bundled.backendPort);
    expect(extra.vscodePort).not.toBe(bundled.vscodePort);
  });

  it("honors OH_CANVAS_EXTRA_BACKEND_PORT and OH_CANVAS_EXTRA_VSCODE_PORT", () => {
    const config = buildExtraBackendConfig(repoRoot, {
      OH_CANVAS_EXTRA_BACKEND_PORT: "29000",
      OH_CANVAS_EXTRA_VSCODE_PORT: "29001",
      OH_SESSION_API_KEY_PATH: isolatedKeyPath(),
    });

    expect(config.backendPort).toBe(29000);
    expect(config.vscodePort).toBe(29001);
    expect(config.backendBaseUrl).toBe("http://127.0.0.1:29000");
  });

  it("shares state dir, conversations, bash events, and secret key with the bundled config", () => {
    const env = {
      OH_CANVAS_SAFE_STATE_DIR: "/tmp/canvas-state",
      OH_SESSION_API_KEY_PATH: isolatedKeyPath(),
    };
    const bundled = buildSafeDevConfig(repoRoot, env);
    const extra = buildExtraBackendConfig(repoRoot, env);

    expect(extra.stateDir).toBe(bundled.stateDir);
    expect(extra.conversationsPath).toBe(bundled.conversationsPath);
    expect(extra.bashEventsDir).toBe(bundled.bashEventsDir);
    expect(extra.tmuxTmpDir).toBe(bundled.tmuxTmpDir);
    expect(extra.secretKey).toBe(bundled.secretKey);
  });

  it("rejects an invalid OH_CANVAS_EXTRA_BACKEND_PORT", () => {
    expect(() =>
      buildExtraBackendConfig(repoRoot, {
        OH_CANVAS_EXTRA_BACKEND_PORT: "not-a-port",
        OH_SESSION_API_KEY_PATH: isolatedKeyPath(),
      }),
    ).toThrow(/Invalid port/);
  });
});
