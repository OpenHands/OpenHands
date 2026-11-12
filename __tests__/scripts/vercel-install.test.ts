import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

const scriptPath = fileURLToPath(
  new URL("../../scripts/vercel-install.sh", import.meta.url),
);
const tempDirs: string[] = [];

function createTempDir() {
  const dir = mkdtempSync(path.join(tmpdir(), "vercel-install-test-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(() => {
  tempDirs
    .splice(0)
    .forEach((dir) => rmSync(dir, { recursive: true, force: true }));
});

describe.skipIf(process.platform === "win32")("Vercel install script", () => {
  it("preserves both GitHub SSH-to-HTTPS rewrites", () => {
    const home = createTempDir();
    const workingDir = createTempDir();
    const binDir = path.join(createTempDir(), "bin");
    mkdirSync(binDir);
    writeFileSync(path.join(binDir, "npm"), "#!/bin/sh\nexit 0\n", {
      mode: 0o755,
    });

    execFileSync("bash", [scriptPath], {
      cwd: workingDir,
      env: { ...process.env, HOME: home, PATH: `${binDir}:${process.env.PATH}` },
    });

    const rewrites = execFileSync(
      "git",
      ["config", "--global", "--get-all", "url.https://github.com/.insteadOf"],
      { env: { ...process.env, HOME: home }, encoding: "utf8" },
    )
      .trim()
      .split("\n");

    expect(rewrites).toEqual(["ssh://git@github.com/", "git@github.com:"]);

    const env = { ...process.env, HOME: home };
    execFileSync("git", ["init", "-q", workingDir], { env });
    for (const remote of [
      "ssh://git@github.com/owner/repo.git",
      "git@github.com:owner/repo.git",
    ]) {
      execFileSync("git", ["-C", workingDir, "remote", "add", "test", remote], {
        env,
      });
      const url = execFileSync(
        "git",
        ["-C", workingDir, "remote", "get-url", "test"],
        { env, encoding: "utf8" },
      );
      expect(url.trim()).toBe("https://github.com/owner/repo.git");
      execFileSync("git", ["-C", workingDir, "remote", "remove", "test"], {
        env,
      });
    }
  });
});
