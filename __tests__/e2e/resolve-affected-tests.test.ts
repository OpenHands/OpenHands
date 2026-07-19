// @vitest-environment node
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

const resolverPath = path.join(
  repoRoot,
  "tests/e2e/mock-llm/scripts/resolve-affected-tests.mjs",
);
const workflowPolicyPath = path.join(
  repoRoot,
  "tests/e2e/mock-llm/scripts/evaluate-workflow-policy.mjs",
);

function resolveAffectedTests(files: string[]) {
  const output = execFileSync(
    process.execPath,
    [resolverPath, "--files", files.join(",")],
    { cwd: repoRoot, encoding: "utf-8" },
  ).trim();

  return output.length > 0 ? output.split(/\s+/) : [];
}

function evaluateWorkflowPolicy() {
  const output = execFileSync(
    process.execPath,
    [workflowPolicyPath, "--json"],
    {
      cwd: repoRoot,
      encoding: "utf-8",
    },
  );

  return JSON.parse(output) as Array<{
    workflow: string;
    scenario: string;
    workflowTriggered: boolean;
    jobRuns: boolean;
    matchesExpected: boolean;
  }>;
}

describe("mock-LLM E2E affected test resolver", () => {
  it("selects mapped source shards plus regressions", () => {
    expect(
      resolveAffectedTests(["src/components/features/settings/llm-form.tsx"]),
    ).toEqual([
      "tests/e2e/mock-llm/regressions",
      "tests/e2e/mock-llm/settings",
    ]);
  });

  it("runs the full suite for cross-cutting source changes", () => {
    expect(resolveAffectedTests(["src/api/agent-server-adapter.ts"])).toEqual([
      "__ALL__",
    ]);
  });

  it("runs the full suite for unmapped source changes", () => {
    expect(resolveAffectedTests(["src/utils/some-new-helper.ts"])).toEqual([
      "__ALL__",
    ]);
  });

  it("selects the containing feature subset for a test-only new spec change", () => {
    expect(
      resolveAffectedTests([
        "tests/e2e/mock-llm/mcp/mock-llm-new-marketplace.spec.ts",
      ]),
    ).toEqual(["tests/e2e/mock-llm/mcp", "tests/e2e/mock-llm/regressions"]);
  });

  it("selects the exact root spec path defensively for misplaced new specs", () => {
    expect(
      resolveAffectedTests(["tests/e2e/mock-llm/mock-llm-new-root.spec.ts"]),
    ).toEqual([
      "tests/e2e/mock-llm/regressions",
      "tests/e2e/mock-llm/mock-llm-new-root.spec.ts",
    ]);
  });

  it.each([
    ["public/favicon.ico"],
    [".github/workflows/mock-llm-e2e.yml"],
    ["tailwind.config.js"],
    ["hero.ts"],
    ["tests/e2e/mock-llm/test-mapping.json"],
  ])("runs the full suite for relevant trigger path %s", (file) => {
    expect(resolveAffectedTests([file])).toEqual(["__ALL__"]);
  });

  it("runs full E2E only for the intended workflow event matrix", () => {
    const rows = evaluateWorkflowPolicy();
    const byScenario = new Map(
      rows.map((row) => [`${row.workflow}:${row.scenario}`, row]),
    );

    expect(rows.every((row) => row.matchesExpected)).toBe(true);
    expect(byScenario.get("mock-llm-e2e:main push")).toMatchObject({
      workflowTriggered: true,
      jobRuns: true,
    });
    expect(
      byScenario.get("mock-llm-e2e:ordinary same-repository PR"),
    ).toMatchObject({
      workflowTriggered: true,
      jobRuns: false,
    });
    expect(byScenario.get("mock-llm-e2e:non-main push")).toMatchObject({
      workflowTriggered: false,
      jobRuns: false,
    });
    expect(
      byScenario.get("mock-llm-e2e:same-repository release-please PR"),
    ).toMatchObject({ workflowTriggered: true, jobRuns: true });
    expect(
      byScenario.get(
        "mock-llm-docker-e2e:successful Docker workflow_run on main",
      ),
    ).toMatchObject({ workflowTriggered: true, jobRuns: true });
    expect(
      byScenario.get(
        "mock-llm-docker-e2e:successful Docker workflow_run on non-main",
      ),
    ).toMatchObject({ workflowTriggered: false, jobRuns: false });
    expect(
      byScenario.get("mock-llm-docker-e2e:ordinary same-repository PR"),
    ).toMatchObject({ workflowTriggered: true, jobRuns: false });
    expect(
      byScenario.get("mock-llm-docker-e2e:same-repository release-please PR"),
    ).toMatchObject({ workflowTriggered: true, jobRuns: true });
  });
});
