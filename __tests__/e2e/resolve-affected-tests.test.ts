// @vitest-environment node
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
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
const workflowPath = path.join(repoRoot, ".github/workflows/mock-llm-e2e.yml");
const dockerWorkflowPath = path.join(
  repoRoot,
  ".github/workflows/mock-llm-docker-e2e.yml",
);

function resolveAffectedTests(files: string[]) {
  const output = execFileSync(
    process.execPath,
    [resolverPath, "--files", files.join(",")],
    { cwd: repoRoot, encoding: "utf-8" },
  ).trim();

  return output.length > 0 ? output.split(/\s+/) : [];
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

  it("runs full E2E only after changes reach main or on release PRs", () => {
    const workflow = readFileSync(workflowPath, "utf-8");
    const dockerWorkflow = readFileSync(dockerWorkflowPath, "utf-8");
    const workflowTriggers = workflow.slice(
      workflow.indexOf("on:\n"),
      workflow.indexOf("\nconcurrency:"),
    );
    const dockerWorkflowTriggers = dockerWorkflow.slice(
      dockerWorkflow.indexOf("on:\n"),
      dockerWorkflow.indexOf("\nconcurrency:"),
    );

    // The PR trigger preserves the existing required check context. Ordinary
    // PRs skip the job; release-please PRs run the full suite before merge.
    expect(workflowTriggers).toContain("  pull_request:");
    expect(workflowTriggers).toContain("  push:\n    branches: [main]");
    expect(workflow).toContain(
      "startsWith(github.head_ref, 'release-please--branches--')",
    );

    expect(dockerWorkflowTriggers).toContain("  pull_request:");
    expect(dockerWorkflowTriggers).toContain("  workflow_run:");
    expect(dockerWorkflowTriggers).toContain("    branches: [main]");
    expect(dockerWorkflowTriggers).toContain("  workflow_dispatch:");
    expect(dockerWorkflow).toContain(
      "startsWith(github.head_ref, 'release-please--branches--')",
    );
  });
});
