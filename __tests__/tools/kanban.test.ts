// @vitest-environment node
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");

describe("kanban python suite", () => {
  it("passes model and API tests", () => {
    const result = spawnSync("python3", ["tools/test_kanban.py"], {
      cwd: repoRoot,
      encoding: "utf8",
    });
    expect(result.status, `${result.stdout}\n${result.stderr}`).toBe(0);
  });
});
