// @vitest-environment node
//
// The automation KV secret signs per-run KV tokens and encrypts automation KV
// state at rest. It used to default to the session API key, so rotating
// LOCAL_BACKEND_API_KEY silently changed the encryption key and left every
// stored KV document undecryptable (OpenHands#17424).
//
// The npm launcher's half is covered in dev-with-automation.test.ts against
// the real function. This file covers the Docker half: the entrypoint has no
// importable surface, so the resolution block is extracted between its markers
// and executed under bash, exercising the shipped precedence rather than
// asserting that particular strings appear in the file.
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

const entrypoint = readFileSync(
  path.join(repoRoot, "docker/entrypoint.sh"),
  "utf-8",
);

const BLOCK_START = "# >>> automation-kv-secret";
const BLOCK_END = "# <<< automation-kv-secret";

function kvSecretBlock(): string {
  const start = entrypoint.indexOf(BLOCK_START);
  const end = entrypoint.indexOf(BLOCK_END);
  if (start === -1 || end === -1) {
    throw new Error(
      `docker/entrypoint.sh is missing the "${BLOCK_START}"/"${BLOCK_END}" markers; ` +
        "the KV secret block can no longer be located, so its behavior is untested.",
    );
  }
  return entrypoint.slice(start, end);
}

interface ResolvedKvSecret {
  status: number | null;
  stderr: string;
  secret: string;
}

function resolveKvSecret(
  stateDir: string,
  env: Record<string, string> = {},
): ResolvedKvSecret {
  const script = [
    "set -uo pipefail",
    // Defined near the top of entrypoint.sh, above the extracted block.
    `log() { printf '%s\\n' "$*" >&2; }`,
    `STATE_DIR="${stateDir}"`,
    kvSecretBlock(),
    `printf '%s\\n' "$AUTOMATION_KV_SECRET"`,
  ].join("\n");

  // Deliberately not inheriting the ambient environment: a developer with
  // AUTOMATION_KV_SECRET exported would otherwise change what these measure.
  const res = spawnSync("bash", ["-c", script], {
    encoding: "utf-8",
    env: { PATH: process.env.PATH ?? "", ...env },
  });
  return {
    status: res.status,
    stderr: res.stderr,
    secret: res.stdout.trim(),
  };
}

describe("docker automation KV secret", () => {
  const dirs: string[] = [];

  afterEach(() => {
    while (dirs.length > 0) {
      const dir = dirs.pop();
      if (dir) rmSync(dir, { recursive: true, force: true });
    }
  });

  function makeStateDir(): string {
    const dir = mkdtempSync(path.join(tmpdir(), "docker-kv-secret-"));
    dirs.push(dir);
    return dir;
  }

  function secretFile(stateDir: string): string {
    return path.join(stateDir, "automation-kv-secret.txt");
  }

  it("keeps the KV secret stable when the session API key is rotated", () => {
    const stateDir = makeStateDir();

    const before = resolveKvSecret(stateDir, {
      EFFECTIVE_SESSION_KEY: "key-a",
    });
    // A restart of the container with a rotated LOCAL_BACKEND_API_KEY.
    const after = resolveKvSecret(stateDir, { EFFECTIVE_SESSION_KEY: "key-b" });

    expect(before.status).toBe(0);
    expect(after.status).toBe(0);
    expect(after.secret).toBe(before.secret);
  });

  // Images upgrading from the old session-key default already hold KV state
  // encrypted under the session key in use at that moment; seeding the file
  // with it keeps that state readable.
  it("seeds the persisted file with the effective session key", () => {
    const stateDir = makeStateDir();

    const resolved = resolveKvSecret(stateDir, {
      EFFECTIVE_SESSION_KEY: "key-a",
    });

    expect(resolved.secret).toBe("key-a");
    expect(readFileSync(secretFile(stateDir), "utf-8")).toBe("key-a");
  });

  it("prefers an explicit AUTOMATION_KV_SECRET and leaves the file alone", () => {
    const stateDir = makeStateDir();

    const resolved = resolveKvSecret(stateDir, {
      EFFECTIVE_SESSION_KEY: "key-a",
      AUTOMATION_KV_SECRET: "explicit-kv-secret",
    });

    expect(resolved.secret).toBe("explicit-kv-secret");
    expect(() => readFileSync(secretFile(stateDir), "utf-8")).toThrow();
  });

  it("reuses an already persisted secret instead of the session key", () => {
    const stateDir = makeStateDir();
    writeFileSync(secretFile(stateDir), "previously-persisted");

    expect(
      resolveKvSecret(stateDir, { EFFECTIVE_SESSION_KEY: "key-a" }).secret,
    ).toBe("previously-persisted");
  });
});
