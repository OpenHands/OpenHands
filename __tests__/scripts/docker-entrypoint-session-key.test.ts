// @vitest-environment node
//
// Regression test for #17763: the Docker entrypoint must mirror the resolved
// session key into OH_SESSION_API_KEYS_0 even when the user supplied
// LOCAL_BACKEND_API_KEY directly. The agent server reads
// OH_SESSION_API_KEYS_0, not LOCAL_BACKEND_API_KEY, so before this fix a
// custom LOCAL_BACKEND_API_KEY left the agent server with no session keys
// at all. Like the npm launcher (scripts/dev-safe.mjs), the entrypoint
// deliberately does not set a SESSION_API_KEY alias.
//
// The test executes the actual key-resolution section of the shipped
// docker/entrypoint.sh under bash (extracted at test time, so it tracks the
// shipped file), with each scenario run in a clean environment.
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

function extractKeyResolutionSection(): string {
  const src = readFileSync(
    path.join(repoRoot, "docker/entrypoint.sh"),
    "utf-8",
  );
  const startAnchor =
    'if [ -z "${LOCAL_BACKEND_API_KEY:-}" ] && [ -z "${OH_SESSION_API_KEYS_0:-}" ]; then';
  const start = src.indexOf(startAnchor);
  expect(start).toBeGreaterThanOrEqual(0);
  const endAnchor = "export AUTOMATION_KV_SECRET=";
  const end = src.indexOf(endAnchor, start);
  expect(end).toBeGreaterThan(start);
  const endOfLine = src.indexOf("\n", end);
  return src.slice(start, endOfLine + 1);
}

const section = extractKeyResolutionSection();

interface Resolved {
  oh: string;
  automation: string;
}

// Runs the shipped section in a clean bash with only the given env vars and
// reports what it exported.
function resolveKeys(env: Record<string, string>): Resolved {
  const script = [
    "set -u",
    "log() { :; }",
    section,
    'printf "OH=%s\\nAUTO=%s\\n" "${OH_SESSION_API_KEYS_0:-}" "${OPENHANDS_AUTOMATION_API_KEY:-}"',
  ].join("\n");
  const r = spawnSync("bash", ["-c", script], {
    env: { PATH: process.env.PATH ?? "/usr/bin:/bin", ...env },
    encoding: "utf-8",
  });
  expect(r.status).toBe(0);
  const out = Object.fromEntries(
    r.stdout
      .trim()
      .split("\n")
      .map((line) => line.split("=", 2) as [string, string]),
  );
  return { oh: out.OH, automation: out.AUTO };
}

describe("docker/entrypoint.sh session key resolution (#17763)", () => {
  it("mirrors a user-supplied LOCAL_BACKEND_API_KEY into the agent-server keys", () => {
    const r = resolveKeys({
      LOCAL_BACKEND_API_KEY: "test-custom-key",
      API_KEY_FILE: "/nonexistent/api-key.txt",
    });
    expect(r.oh).toBe("test-custom-key");
    expect(r.automation).toBe("test-custom-key");
  });

  it("an explicit OH_SESSION_API_KEYS_0 wins over LOCAL_BACKEND_API_KEY", () => {
    const r = resolveKeys({
      LOCAL_BACKEND_API_KEY: "local-key",
      OH_SESSION_API_KEYS_0: "oh-key",
      API_KEY_FILE: "/nonexistent/api-key.txt",
    });
    expect(r.oh).toBe("oh-key");
    expect(r.automation).toBe("oh-key");
  });

  it("mirrors a freshly generated key when nothing is supplied", () => {
    // No env key and no persisted file: the entrypoint generates a key.
    // The generated value is random, so assert shape and consistency.
    const dir = mkdtempSync(path.join(tmpdir(), "entrypoint-gen-"));
    const r = resolveKeys({ API_KEY_FILE: path.join(dir, "api-key.txt") });
    expect(r.oh).toMatch(/^[0-9a-f]{64}$/);
    expect(r.automation).toBe(r.oh);
  });

  it("falls back to the persisted key file when no env key is supplied", () => {
    const dir = mkdtempSync(path.join(tmpdir(), "entrypoint-key-"));
    const keyFile = path.join(dir, "api-key.txt");
    writeFileSync(keyFile, "persisted-key", { mode: 0o600 });
    const r = resolveKeys({ API_KEY_FILE: keyFile });
    expect(r.oh).toBe("persisted-key");
    expect(r.automation).toBe("persisted-key");
  });
});
