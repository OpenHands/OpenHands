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
import {
  chmodSync,
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
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

// Include all preflight blocks, ending before secret-key initialization.
function extractPreflight(): string {
  const start = entrypoint.indexOf("report_unwritable() {");
  const end = entrypoint.indexOf("# OH_SECRET_KEY is required", start);
  if (start < 0 || end < 0) {
    throw new Error(
      "could not locate the preflight region in docker/entrypoint.sh",
    );
  }
  return entrypoint.slice(start, end);
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
      HOME: path.join(scratch, "home"),
      XDG_CONFIG_HOME: path.join(scratch, "config"),
      OPENHANDS_DIR: writable,
      OH_PERSISTENCE_DIR: writable,
      OH_CONVERSATIONS_PATH: writable,
      OH_BASH_EVENTS_DIR: writable,
      LOCAL_STORAGE_PATH: writable,
      AUTOMATION_WORKSPACE_BASE: writable,
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
    "preserves an existing .write-test file",
    () => {
      const directory = path.join(scratch, "preserved");
      const setup = spawnSync("mkdir", ["-p", directory]);
      expect(setup.status).toBe(0);
      const existing = path.join(directory, ".write-test");
      writeFileSync(existing, "user data");
      const result = runScript("", { OPENHANDS_DIR: directory });
      expect(result.status).toBe(0);
      expect(readFileSync(existing, "utf-8")).toBe("user data");
    },
  );

  it.skipIf(process.platform === "win32").each(["default", "override"])(
    "creates and probes the %s browser config directory",
    (mode) => {
      const home = path.join(scratch, `browser-home-${mode}`);
      const config = path.join(scratch, `browser-config-${mode}`);
      const expected = mode === "default" ? path.join(home, ".config") : config;

      const result = runScript("", {
        HOME: home,
        XDG_CONFIG_HOME: mode === "default" ? "" : config,
      });

      expect(result.status).toBe(0);
      expect(existsSync(expected)).toBe(true);
      expect(existsSync(path.join(expected, ".write-test"))).toBe(false);
      if (mode === "override") {
        expect(existsSync(path.join(home, ".config"))).toBe(false);
      }
    },
  );

  it.skipIf(process.platform === "win32").each(["default", "override"])(
    "fails before startup when the %s browser config path is inaccessible",
    (mode) => {
      const blocker = path.join(scratch, `browser-blocker-${mode}`);
      writeFileSync(blocker, "");
      const home = mode === "default" ? blocker : path.join(scratch, "home");
      const config = path.join(blocker, ".config");

      const result = runScript("printf 'STARTED_BACKENDS'", {
        HOME: home,
        XDG_CONFIG_HOME: mode === "default" ? "" : config,
      });

      expect(result.status).toBe(1);
      expect(result.stdout).not.toContain("STARTED_BACKENDS");
      expect(result.stderr).toContain(
        `Browser config directory is not writable: ${config}`,
      );
      expect(result.stderr).toContain("Underlying error:");
      expect(result.stderr).not.toMatch(/^(mkdir|touch):/m);
    },
  );

  it.skipIf(process.platform === "win32")(
    "suggests a writable config path for browser permission failures",
    () => {
      const result = runScript(
        `report_unwritable "Browser config directory" "/home/openhands/.config" "Permission denied" "~/.config"`,
      );

      expect(result.status).toBe(0);
      expect(result.stderr).toContain(
        "Set XDG_CONFIG_HOME to a writable directory",
      );
      expect(result.stderr).toContain("keep the default openhands user");
      expect(result.stderr).not.toContain("Run ON THE HOST");
    },
  );

  it.skipIf(process.platform === "win32")(
    "suggests private state access and a named volume",
    () => {
      const result = runReport("Permission denied");
      expect(result.stderr).toContain("ownership or ACLs");
      expect(result.stderr).toContain("named Docker volume");
      expect(result.stderr).not.toContain("chmod a+rwX");
    },
  );

  it.skipIf(process.platform === "win32")(
    "explains existing workspace file permissions",
    () => {
      const result = runScript(
        `report_unwritable "Workspace directory" "/projects" "Permission denied" "~/projects"`,
      );
      expect(result.stderr).toContain(
        "Existing project files also need read/write access",
      );
      expect(result.stderr).not.toContain("named Docker volume");
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

  it.skipIf(process.platform === "win32" || process.getuid?.() === 0)(
    "rejects an existing inaccessible automation database before startup",
    () => {
      const db = path.join(scratch, "locked.db");
      writeFileSync(db, "existing state");
      chmodSync(db, 0o400);
      const result = runScript("printf STARTED_BACKENDS", {
        AUTOMATION_DB_FILE: db,
      });
      expect(result.status).toBe(1);
      expect(result.stderr).toContain(
        "Existing automation database is not readable and writable",
      );
      expect(result.stdout).not.toContain("STARTED_BACKENDS");
    },
  );

  it.skipIf(process.platform === "win32")(
    "checks overridden automation storage before startup",
    () => {
      const blocker = path.join(scratch, "storage-blocker");
      writeFileSync(blocker, "");
      const result = runScript("printf STARTED_BACKENDS", {
        LOCAL_STORAGE_PATH: path.join(blocker, "storage"),
      });
      expect(result.status).toBe(1);
      expect(result.stdout).not.toContain("STARTED_BACKENDS");
      expect(result.stderr).toContain("storage-blocker");
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
