// @vitest-environment node
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

describe("lint-staged portability", () => {
  it("runs staged typecheck through Node instead of a POSIX shell", () => {
    const packageJson = JSON.parse(
      readFileSync(path.join(repoRoot, "package.json"), "utf-8"),
    ) as {
      "lint-staged": Record<string, string[]>;
    };

    const commands = Object.values(packageJson["lint-staged"]).flat();

    expect(commands).not.toContain("bash -c 'npm run typecheck:staged'");
    expect(commands).toContain("node scripts/run-staged-typecheck.mjs");
  });

  it("uses npm's JS CLI on Windows instead of spawning npm.cmd directly", () => {
    const source = readFileSync(
      path.join(repoRoot, "scripts", "run-staged-typecheck.mjs"),
      "utf-8",
    );

    expect(source).toContain('"npm-cli.js"');
    expect(source).not.toContain('"npm.cmd"');
    expect(source).toContain("spawnSync");
  });
});
