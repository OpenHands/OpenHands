// @vitest-environment node

import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const entrypoint = readFileSync(
  path.join(repoRoot, "docker/entrypoint.sh"),
  "utf-8",
);
const blockStart = "# >>> docker-session-key-policy";
const blockEnd = "# <<< docker-session-key-policy";

function sessionKeyPolicyBlock(): string {
  const start = entrypoint.indexOf(blockStart);
  const end = entrypoint.indexOf(blockEnd);
  if (start === -1 || end === -1) {
    throw new Error("Docker session-key policy markers are missing");
  }
  return entrypoint.slice(start, end);
}

function resolveStaticServerArgs(allowLanSessionKey: string | undefined): {
  args: string[];
  stderr: string;
} {
  const script = [
    "set -uo pipefail",
    "PORT=8000",
    "log() { printf '%s\\n' \"$*\" >&2; }",
    sessionKeyPolicyBlock(),
    "printf '%s\\n' \"${STATIC_SERVER_SESSION_KEY_ARGS[@]}\"",
  ].join("\n");
  const env: Record<string, string> = { PATH: process.env.PATH ?? "" };
  if (allowLanSessionKey !== undefined) {
    env.AGENT_CANVAS_ALLOW_LAN_SESSION_KEY = allowLanSessionKey;
  }
  const result = spawnSync("bash", ["-c", script], {
    encoding: "utf-8",
    env,
  });
  expect(result.status).toBe(0);
  return {
    args: result.stdout.trim() ? result.stdout.trim().split("\n") : [],
    stderr: result.stderr,
  };
}

describe("Docker session-key injection policy", () => {
  it("does not inject the key by default", () => {
    expect(resolveStaticServerArgs(undefined).args).toEqual([]);
  });

  it("requires an explicit true value and warns when enabled", () => {
    expect(resolveStaticServerArgs("1").args).toEqual([]);

    const enabled = resolveStaticServerArgs("true");
    expect(enabled.args).toEqual(["--allow-lan-session-key"]);
    expect(enabled.stderr).toContain("WARNING");
    expect(enabled.stderr).toContain("host loopback only");
  });
});
