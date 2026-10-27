// @vitest-environment node

import { spawn, spawnSync } from "node:child_process";
import { once } from "node:events";
import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { resolveAutomationKvSecret } from "../../scripts/dev-with-automation.mjs";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const entrypoint = readFileSync(
  path.join(repoRoot, "docker/entrypoint.sh"),
  "utf-8",
);

function kvSecretBlock(): string {
  const start = entrypoint.indexOf("# >>> automation-kv-secret");
  const end = entrypoint.indexOf("# <<< automation-kv-secret");
  if (start === -1 || end === -1) {
    throw new Error("Docker automation-kv-secret markers are missing");
  }
  return entrypoint.slice(start, end);
}

function dockerScript(): string {
  return [
    "set -uo pipefail",
    "log() { :; }",
    "log_error() { printf '%s\\n' \"$*\" >&2; }",
    kvSecretBlock(),
    'printf "%s" "$AUTOMATION_KV_SECRET"',
  ].join("\n");
}

function runDockerKvSecret(env: Record<string, string>) {
  return spawnSync("bash", ["-c", dockerScript()], {
    encoding: "utf-8",
    env: { PATH: process.env.PATH ?? "", ...env },
  });
}

function dockerKvSecret(env: Record<string, string>): string {
  const result = runDockerKvSecret(env);
  expect(result.stderr).toBe("");
  expect(result.status).toBe(0);
  return result.stdout;
}

// Runs a launcher in its own process and resolves to what it printed, so
// several can race to pin the same state dir.
async function racingLauncher(
  launcher: "docker" | "npm",
  stateDir: string,
  sessionKey: string,
): Promise<string> {
  const child =
    launcher === "docker"
      ? spawn("bash", ["-c", dockerScript()], {
          env: {
            PATH: process.env.PATH ?? "",
            STATE_DIR: stateDir,
            EFFECTIVE_SESSION_KEY: sessionKey,
          },
        })
      : spawn(
          process.execPath,
          [
            "--input-type=module",
            "-e",
            [
              `const { resolveAutomationKvSecret } = await import(${JSON.stringify(
                path.join(repoRoot, "scripts/dev-with-automation.mjs"),
              )});`,
              // The launcher logs to stdout too, so the key goes on its own last line.
              `process.stdout.write("\\n" + resolveAutomationKvSecret(${JSON.stringify(
                {
                  stateDir,
                  sessionApiKey: sessionKey,
                },
              )}, {}));`,
            ].join("\n"),
          ],
          { env: { ...process.env, AUTOMATION_KV_SECRET: "" } },
        );
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => (stdout += chunk));
  child.stderr.on("data", (chunk) => (stderr += chunk));
  const [code] = await once(child, "close");
  expect({ code, stderr }).toEqual({ code: 0, stderr: "" });
  return stdout.split("\n").at(-1) ?? "";
}

// A directory of stand-ins for coreutils commands that always fail, put
// first on PATH so the entrypoint block sees that step fail.
function failingCommands(dir: string, ...names: string[]): string {
  const bin = path.join(dir, "failing-bin");
  mkdirSync(bin);
  for (const name of names) {
    writeFileSync(path.join(bin, name), "#!/bin/sh\nexit 1\n", {
      mode: 0o755,
    });
  }
  return `${bin}:${process.env.PATH ?? ""}`;
}

const isPosix = process.platform !== "win32";
const pinMode = (file: string) => statSync(file).mode & 0o777;

describe("Docker automation KV secret", () => {
  let stateDir = "";

  afterEach(() => {
    if (stateDir) rmSync(stateDir, { recursive: true, force: true });
  });

  function freshStateDir(): string {
    stateDir = mkdtempSync(path.join(tmpdir(), "docker-kv-secret-"));
    return stateDir;
  }

  it("keeps the first session key as the KV secret after the session key rotates", () => {
    const dir = freshStateDir();

    const first = dockerKvSecret({
      STATE_DIR: dir,
      EFFECTIVE_SESSION_KEY: "session-key-a",
    });
    const afterRotation = dockerKvSecret({
      STATE_DIR: dir,
      EFFECTIVE_SESSION_KEY: "session-key-b",
    });

    expect(first).toBe("session-key-a");
    expect(afterRotation).toBe("session-key-a");
  });

  it("prefers an explicit AUTOMATION_KV_SECRET", () => {
    const dir = freshStateDir();

    expect(
      dockerKvSecret({
        STATE_DIR: dir,
        EFFECTIVE_SESSION_KEY: "session-key-a",
        AUTOMATION_KV_SECRET: "explicit-secret",
      }),
    ).toBe("explicit-secret");
  });

  it.each([
    ["a trailing newline", "pinned-key\n", "pinned-key"],
    ["a Windows line ending", "pinned-key\r\n", "pinned-key"],
    ["several trailing newlines", "pinned-key\n\n", "pinned-key"],
    ["only a newline", "\n", "session-key"],
    ["only spaces", "   ", "session-key"],
  ])(
    "reads a hand-edited pin with %s the same way as the npm launcher",
    (_, content, expected) => {
      const dockerDir = freshStateDir();
      const npmDir = mkdtempSync(path.join(tmpdir(), "npm-kv-secret-"));

      try {
        writeFileSync(
          path.join(dockerDir, "automation-kv-secret.txt"),
          content,
        );
        writeFileSync(path.join(npmDir, "automation-kv-secret.txt"), content);

        // A blank pin is replaced on the first launch, so the second launch,
        // with a rotated session key, must still read the same value.
        for (const sessionKey of ["session-key", "rotated"]) {
          expect(
            dockerKvSecret({
              STATE_DIR: dockerDir,
              EFFECTIVE_SESSION_KEY: sessionKey,
            }),
          ).toBe(expected);
          expect(
            resolveAutomationKvSecret(
              { stateDir: npmDir, sessionApiKey: sessionKey },
              {},
            ),
          ).toBe(expected);
        }
      } finally {
        rmSync(npmDir, { recursive: true, force: true });
      }
    },
  );

  it("shares the pinned secret with the npm launcher", () => {
    const dockerFirst = freshStateDir();
    dockerKvSecret({ STATE_DIR: dockerFirst, EFFECTIVE_SESSION_KEY: "docker" });
    const npmFirst = mkdtempSync(path.join(tmpdir(), "npm-kv-secret-"));

    try {
      resolveAutomationKvSecret(
        { stateDir: npmFirst, sessionApiKey: "npm" },
        {},
      );

      expect(
        resolveAutomationKvSecret(
          { stateDir: dockerFirst, sessionApiKey: "rotated" },
          {},
        ),
      ).toBe("docker");
      expect(
        dockerKvSecret({
          STATE_DIR: npmFirst,
          EFFECTIVE_SESSION_KEY: "rotated",
        }),
      ).toBe("npm");
    } finally {
      rmSync(npmFirst, { recursive: true, force: true });
    }
  });

  describe.each(["docker", "npm"] as const)(
    "pin file contract (%s launcher)",
    (launcher) => {
      function resolve(dir: string, sessionKey: string): string {
        return launcher === "docker"
          ? dockerKvSecret({
              STATE_DIR: dir,
              EFFECTIVE_SESSION_KEY: sessionKey,
            })
          : resolveAutomationKvSecret(
              { stateDir: dir, sessionApiKey: sessionKey },
              {},
            );
      }

      function expectRefused(dir: string, message: RegExp) {
        if (launcher === "docker") {
          const result = runDockerKvSecret({
            STATE_DIR: dir,
            EFFECTIVE_SESSION_KEY: "session-key",
          });
          expect(result.status).toBe(1);
          expect(result.stdout).toBe("");
          expect(result.stderr).toMatch(message);
        } else {
          expect(() =>
            resolveAutomationKvSecret(
              { stateDir: dir, sessionApiKey: "session-key" },
              {},
            ),
          ).toThrow(message);
        }
      }

      it.runIf(isPosix)("creates the pin with mode 0600", () => {
        const dir = freshStateDir();
        resolve(dir, "session-key");
        expect(pinMode(path.join(dir, "automation-kv-secret.txt"))).toBe(0o600);
      });

      it.runIf(isPosix)(
        "makes a permissive pin private before using it",
        () => {
          const dir = freshStateDir();
          const pin = path.join(dir, "automation-kv-secret.txt");
          writeFileSync(pin, "pinned-key\n");
          chmodSync(pin, 0o644);

          expect(resolve(dir, "session-key")).toBe("pinned-key");
          expect(pinMode(pin)).toBe(0o600);
        },
      );

      it.runIf(isPosix)(
        "makes a permissive blank pin private when it receives the key",
        () => {
          const dir = freshStateDir();
          const pin = path.join(dir, "automation-kv-secret.txt");
          writeFileSync(pin, "\n");
          chmodSync(pin, 0o644);

          expect(resolve(dir, "session-key")).toBe("session-key");
          expect(readFileSync(pin, "utf-8").trim()).toBe("session-key");
          expect(pinMode(pin)).toBe(0o600);
        },
      );

      it("refuses a pin that is a symbolic link", () => {
        const dir = freshStateDir();
        const target = path.join(dir, "elsewhere.txt");
        writeFileSync(target, "pinned-key\n");
        symlinkSync(target, path.join(dir, "automation-kv-secret.txt"));

        expectRefused(dir, /symbolic link/);
        expect(readFileSync(target, "utf-8")).toBe("pinned-key\n");
      });

      it("refuses a pin that is not a regular file", () => {
        const dir = freshStateDir();
        mkdirSync(path.join(dir, "automation-kv-secret.txt"));

        expectRefused(dir, /not a regular file/);
      });

      it.runIf(isPosix)("refuses a pin that is a FIFO without blocking", () => {
        const dir = freshStateDir();
        const made = spawnSync("mkfifo", [
          path.join(dir, "automation-kv-secret.txt"),
        ]);
        expect(made.status).toBe(0);

        expectRefused(dir, /not a regular file/);
      });

      it("refuses a state dir that cannot be created", () => {
        const dir = freshStateDir();
        writeFileSync(path.join(dir, "not-a-dir"), "");

        expectRefused(path.join(dir, "not-a-dir", "state"), /not-a-dir/);
      });

      it("gives every concurrent first launch the one pinned key", async () => {
        const dir = freshStateDir();

        const keys = await Promise.all(
          Array.from({ length: 12 }, (_, i) =>
            racingLauncher(launcher, dir, `session-key-${i}`),
          ),
        );

        const pinned = readFileSync(
          path.join(dir, "automation-kv-secret.txt"),
          "utf-8",
        ).replace(/[\r\n]+$/, "");
        expect(new Set(keys)).toEqual(new Set([pinned]));
      });

      it("gives every concurrent launch on a blank pin the one pinned key", async () => {
        // The race is timing-dependent, so run several rounds.
        for (let round = 0; round < 10; round += 1) {
          const dir = freshStateDir();
          const pin = path.join(dir, "automation-kv-secret.txt");
          writeFileSync(pin, "\n", { mode: 0o600 });

          const keys = await Promise.all(
            Array.from({ length: 6 }, (_, i) =>
              racingLauncher(launcher, dir, `session-key-${i}`),
            ),
          );

          const pinned = readFileSync(pin, "utf-8").replace(/[\r\n]+$/, "");
          expect(new Set(keys)).toEqual(new Set([pinned]));
          expect(existsSync(`${pin}.lock`)).toBe(false);
          rmSync(dir, { recursive: true, force: true });
        }
      });
    },
  );

  describe("Docker persistence failures stop the start", () => {
    it.each([
      ["the state directory cannot be created", ["mkdir"], /Cannot create/],
      ["the pin cannot be written", ["mktemp"], /Cannot write/],
      ["the pin cannot be linked into place", ["ln"], /Cannot create/],
    ])("when %s", (_, commands, message) => {
      const dir = freshStateDir();
      const stateDir = path.join(dir, "state");

      const result = runDockerKvSecret({
        PATH: failingCommands(dir, ...commands),
        STATE_DIR: stateDir,
        EFFECTIVE_SESSION_KEY: "session-key",
      });

      expect(result.status).toBe(1);
      expect(result.stdout).toBe("");
      expect(result.stderr).toMatch(message);
    });

    it.runIf(isPosix)("when a permissive pin cannot be made private", () => {
      const dir = freshStateDir();
      const pin = path.join(dir, "automation-kv-secret.txt");
      writeFileSync(pin, "pinned-key\n");
      chmodSync(pin, 0o644);

      const result = runDockerKvSecret({
        PATH: failingCommands(dir, "chmod"),
        STATE_DIR: dir,
        EFFECTIVE_SESSION_KEY: "session-key",
      });

      expect(result.status).toBe(1);
      expect(result.stdout).toBe("");
      expect(result.stderr).toMatch(/chmod 600/);
    });
  });
});
