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

  it("passes agent session linking tests", () => {
    const result = spawnSync("python3", ["tools/test_kanban_agent.py"], {
      cwd: repoRoot,
      encoding: "utf8",
    });
    expect(result.status, `${result.stdout}\n${result.stderr}`).toBe(0);
  });

  it("passes project config tests", () => {
    const result = spawnSync("python3", ["tools/test_project_config.py"], {
      cwd: repoRoot,
      encoding: "utf8",
    });
    expect(result.status, `${result.stdout}\n${result.stderr}`).toBe(0);
  });

  it("passes project bootstrap tests", () => {
    const result = spawnSync("python3", ["tools/test_project_bootstrap.py"], {
      cwd: repoRoot,
      encoding: "utf8",
    });
    expect(result.status, `${result.stdout}\n${result.stderr}`).toBe(0);
  });

  it("passes cost estimator tests", () => {
    const result = spawnSync("python3", ["tools/test_cost_estimator.py"], {
      cwd: repoRoot,
      encoding: "utf8",
    });
    expect(result.status, `${result.stdout}\n${result.stderr}`).toBe(0);
  });
});
