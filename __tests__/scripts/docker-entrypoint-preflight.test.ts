// @vitest-environment node
//
// Regression coverage for the Docker entrypoint's startup preflight.
//
// Two review findings on PR #16512 are pinned here:
//
//   1. It used to diagnose *every* write failure as a uid-10001 permission
//      mismatch, so a read-only mount or a full disk was answered with advice
//      that could not possibly help. The remediation is now selected from the
//      real OS error.
//   2. `2>/dev/null` only scoped to the `touch`, so a failing `mkdir -p` leaked
//      a raw, unprefixed OS error line instead of the curated message.
//
// These tests execute the preflight region lifted out of the real script, so
// they exercise the branch logic rather than a copy of it.
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

const entrypoint = readFileSync(
  path.join(repoRoot, "docker/entrypoint.sh"),
  "utf-8",
);

// `report_unwritable` plus both preflight blocks, ending at the `fi` that
// closes the optional /projects block.
function extractPreflight(): string {
  const start = entrypoint.indexOf("report_unwritable() {");
  const projects = entrypoint.indexOf("if [ -d /projects ]; then");
  const closing = entrypoint.indexOf("\nfi\n", projects);
  if (start < 0 || projects < 0 || closing < 0) {
    throw new Error(
      "could not locate the preflight region in docker/entrypoint.sh",
    );
  }
  return entrypoint.slice(start, closing + "\nfi\n".length);
}

const preflight = extractPreflight();

// Stand-in for the entrypoint's logger. The prefix must match the real one so
// the "no unprefixed line leaks" assertion stays meaningful.
const HARNESS = [
  "set -uo pipefail",
  "log_error() { printf '[agent-canvas] ERROR: %s\\n' \"$*\" >&2; }",
  preflight,
].join("\n");

const scratch = mkdtempSync(path.join(tmpdir(), "oh-preflight-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

function quote(value: string): string {
  return `'${value.replace(/'/g, "'\\''")}'`;
}

// The preflight blocks run before `body`, so the persistence dirs default to a
// writable scratch path that lets those blocks pass.
function runScript(body: string, env: NodeJS.ProcessEnv = {}) {
  const writable = path.join(scratch, "ok");
  return spawnSync("bash", ["-c", `${HARNESS}\n${body}\n`], {
    encoding: "utf-8",
    env: {
      ...process.env,
      OPENHANDS_DIR: writable,
      OH_PERSISTENCE_DIR: writable,
      OH_CONVERSATIONS_PATH: writable,
      OH_BASH_EVENTS_DIR: writable,
      ...env,
    },
  });
}

// Drive the reporter directly with the error strings the kernel actually emits.
function runReport(osError: string) {
  return runScript(
    `report_unwritable "Persistence directory" "/host/.openhands" ${quote(osError)} "~/.openhands"`,
  );
}

describe("docker/entrypoint.sh startup preflight", () => {
  it.skipIf(process.platform === "win32")(
    "surfaces the OS error and suggests the uid fix for a permission denial",
    () => {
      const osError =
        "mkdir: cannot create directory '/host/.openhands': Permission denied";

      const result = runReport(osError);

      expect(result.status).toBe(0);
      expect(result.stderr).toContain(`Underlying error: ${osError}`);
      expect(result.stderr).toContain("chmod a+rwX");
      expect(result.stderr).toContain("--user");
    },
  );

  it.skipIf(process.platform === "win32")(
    "points at the read-only mount instead of a uid mismatch",
    () => {
      const result = runReport(
        "touch: cannot touch '/host/.openhands/.write-test': Read-only file system",
      );

      expect(result.stderr).toContain("Read-only file system");
      expect(result.stderr).toContain("read-only");
      expect(result.stderr).not.toContain("chmod a+rwX");
      expect(result.stderr).not.toContain("--user");
    },
  );

  it.skipIf(process.platform === "win32")(
    "does not invent a remediation for an unrelated filesystem error",
    () => {
      const result = runReport(
        "touch: cannot touch '/host/.openhands/.write-test': No space left on device",
      );

      expect(result.stderr).toContain("No space left on device");
      expect(result.stderr).not.toContain("chmod a+rwX");
      expect(result.stderr).not.toContain("--user");
    },
  );

  it.skipIf(process.platform === "win32")(
    "captures a failing mkdir instead of leaking its raw error line",
    () => {
      // `blocker` is a regular file, so `mkdir -p blocker/.openhands` fails with
      // ENOTDIR before the touch ever runs — precisely the case that a
      // `2>/dev/null` scoped to the touch alone used to miss.
      const blocker = path.join(scratch, "blocker");
      writeFileSync(blocker, "");
      const unwritable = path.join(blocker, ".openhands");

      const result = runScript("", {
        OPENHANDS_DIR: unwritable,
        OH_PERSISTENCE_DIR: unwritable,
        OH_CONVERSATIONS_PATH: unwritable,
        OH_BASH_EVENTS_DIR: unwritable,
      });

      expect(result.status).toBe(1);
      const lines = result.stderr.split("\n").filter((line) => line !== "");
      expect(lines.some((line) => /^(mkdir|touch):/.test(line))).toBe(false);
      expect(result.stderr).toContain("Underlying error:");
      expect(result.stderr).toContain(unwritable);
    },
  );
});
