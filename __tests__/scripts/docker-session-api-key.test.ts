// @vitest-environment node
//
// Regression test for session API key resolution in the Docker entrypoint.
//
// Invariants enforced:
// 1. When LOCAL_BACKEND_API_KEY is provided, OH_SESSION_API_KEYS_0 is exported
//    with that value so agent-server authenticates properly.
// 2. When OH_SESSION_API_KEYS_0 is already set, its value takes precedence over
//    LOCAL_BACKEND_API_KEY.
// 3. When neither is provided, a key is auto-generated, persisted to api-key.txt,
//    and exported as OH_SESSION_API_KEYS_0.
// 4. In all combinations, set -uo pipefail does not trigger an unbound-variable error.
// 5. Downstream automation env vars resolve to the effective session key.
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

function read(rel: string): string {
  return readFileSync(path.join(repoRoot, rel), "utf-8");
}

const entrypoint = read("docker/entrypoint.sh");

const BLOCK_START = "# >>> session-api-key";
const BLOCK_END = "# <<< session-api-key";

function sessionApiKeyBlock(): string {
  const start = entrypoint.indexOf(BLOCK_START);
  const end = entrypoint.indexOf(BLOCK_END);
  if (start === -1 || end === -1) {
    throw new Error(
      `docker/entrypoint.sh is missing the "${BLOCK_START}"/"${BLOCK_END}" markers; ` +
        "the session-api-key block can no longer be located, so its behavior is untested.",
    );
  }
  return entrypoint.slice(start, end);
}

interface ResolvedSessionApiKey {
  status: number | null;
  stdout: string;
  stderr: string;
  sessionApiKey: string;
  effectiveSessionKey: string;
  automationApiKey: string;
}

describe.skipIf(process.platform === "win32")(
  "docker session api key resolution",
  () => {
    let tmpDir: string;

    beforeEach(() => {
      tmpDir = mkdtempSync(path.join(tmpdir(), "session-key-test-"));
    });

    afterEach(() => {
      rmSync(tmpDir, { recursive: true, force: true });
    });

    function resolveSessionApiKey(
      env: Record<string, string | undefined> = {},
    ): ResolvedSessionApiKey {
      const script = [
        "set -uo pipefail",
        `STATE_DIR="${tmpDir}"`,
        `log() { :; }`,
        `log_error() { printf 'ERROR: %s\\n' "$*" >&2; }`,
        sessionApiKeyBlock(),
        `EFFECTIVE_SESSION_KEY="\${OH_SESSION_API_KEYS_0:-\${LOCAL_BACKEND_API_KEY:-}}"`,
        `OPENHANDS_AUTOMATION_API_KEY="\${OPENHANDS_AUTOMATION_API_KEY:-\${EFFECTIVE_SESSION_KEY}}"`,
        `printf '%s\\n%s\\n%s\\n' "\${OH_SESSION_API_KEYS_0:-}" "\${EFFECTIVE_SESSION_KEY:-}" "\${OPENHANDS_AUTOMATION_API_KEY:-}"`,
      ].join("\n");

      const res = spawnSync("bash", ["-c", script], {
        encoding: "utf-8",
        env: { PATH: process.env.PATH ?? "", ...env } as Record<string, string>,
      });

      const [sessionApiKey = "", effectiveSessionKey = "", automationApiKey = ""] =
        res.stdout.trim().split("\n");

      return {
        status: res.status,
        stdout: res.stdout,
        stderr: res.stderr,
        sessionApiKey,
        effectiveSessionKey,
        automationApiKey,
      };
    }

    it("exports OH_SESSION_API_KEYS_0 when LOCAL_BACKEND_API_KEY is supplied", () => {
      const resolved = resolveSessionApiKey({
        LOCAL_BACKEND_API_KEY: "my-custom-key",
      });
      expect(resolved.status).toBe(0);
      expect(resolved.sessionApiKey).toBe("my-custom-key");
      expect(resolved.effectiveSessionKey).toBe("my-custom-key");
      expect(resolved.automationApiKey).toBe("my-custom-key");
    });

    it("preserves OH_SESSION_API_KEYS_0 precedence when both are set", () => {
      const resolved = resolveSessionApiKey({
        LOCAL_BACKEND_API_KEY: "user-backend-key",
        OH_SESSION_API_KEYS_0: "upstream-session-key",
      });
      expect(resolved.status).toBe(0);
      expect(resolved.sessionApiKey).toBe("upstream-session-key");
      expect(resolved.effectiveSessionKey).toBe("upstream-session-key");
    });

    it("auto-generates and persists a key when neither is set", () => {
      const resolved = resolveSessionApiKey();
      expect(resolved.status).toBe(0);
      expect(resolved.sessionApiKey).toBeTruthy();
      expect(resolved.effectiveSessionKey).toBe(resolved.sessionApiKey);

      // Re-running with same STATE_DIR reuses persisted key
      const secondRun = resolveSessionApiKey();
      expect(secondRun.status).toBe(0);
      expect(secondRun.sessionApiKey).toBe(resolved.sessionApiKey);
    });

    it("completes startup without unbound-variable failure in all combinations", () => {
      const combinations: Record<string, string | undefined>[] = [
        {},
        { LOCAL_BACKEND_API_KEY: "test-key" },
        { OH_SESSION_API_KEYS_0: "test-key-0" },
        { LOCAL_BACKEND_API_KEY: "key-a", OH_SESSION_API_KEYS_0: "key-b" },
      ];
      for (const env of combinations) {
        const resolved = resolveSessionApiKey(env);
        expect(resolved.status).toBe(0);
        expect(resolved.stderr).toBe("");
      }
    });
  },
);
