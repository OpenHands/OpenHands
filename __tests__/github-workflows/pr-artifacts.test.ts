// @vitest-environment node
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

function read(rel: string): string {
  return readFileSync(path.join(repoRoot, rel), "utf-8");
}

describe("pr-artifacts workflow and documentation", () => {
  it("does not trigger or clean up on PR approval in pr-artifacts.yml", () => {
    const workflow = read(".github/workflows/pr-artifacts.yml");

    expect(workflow).not.toContain("pull_request_review");
    expect(workflow).not.toContain("cleanup-on-approval");
    expect(workflow).toContain("check-pr-artifacts:");
    expect(workflow).toContain("cleanup-after-merge:");
  });

  it("posts a uniform notice stating .pr/ is removed after merge to main via cleanup PR", () => {
    const workflow = read(".github/workflows/pr-artifacts.yml");

    expect(workflow).not.toContain("isFork");
    expect(workflow).not.toContain(
      "The directory will be **automatically removed when the PR is approved**.",
    );
    expect(workflow).toContain(
      "The workflow will **open or update a cleanup PR against `main` after merge**",
    );
  });

  it("updates docs/DEVELOPMENT.md to describe post-merge cleanup via cleanup PR", () => {
    const doc = read("docs/DEVELOPMENT.md");

    expect(doc).not.toContain("removed automatically when the PR is approved");
    expect(doc).toContain("cleanup PR");
    expect(doc).toContain("commit SHA");
  });

  it("updates .github/pull_request_template.md to describe post-merge cleanup", () => {
    const template = read(".github/pull_request_template.md");

    expect(template).not.toContain("when a same-repository PR is approved");
    expect(template).toContain("cleanup PR");
    expect(template).toContain("commit SHA");
  });

  it("updates .agents/skills/pr-design-doc/SKILL.md to describe post-merge cleanup", () => {
    const skill = read(".agents/skills/pr-design-doc/SKILL.md");

    expect(skill).not.toContain("removes `.pr/` after approval");
    expect(skill).not.toContain("after approval");
    expect(skill).toContain("cleanup PR");
  });

  it("updates .agents/skills/pr-design-doc/references/html-craft.md to describe post-merge cleanup", () => {
    const craft = read(".agents/skills/pr-design-doc/references/html-craft.md");

    expect(craft).not.toContain("when `.pr/` is removed after approval");
    expect(craft).toContain("cleanup PR");
    expect(craft).toContain("commit-pinned link");
  });

  it("updates .agents/skills/e2e-testing/references/guide.md to describe post-merge cleanup", () => {
    const guide = read(".agents/skills/e2e-testing/references/guide.md");

    expect(guide).not.toContain(
      "removes them after PR approval for same-repo PRs",
    );
    expect(guide).toContain("cleanup PR against `main`");
  });

  it("updates .agents/skills/custom-codereview-guide.md to describe post-merge cleanup", () => {
    const reviewGuide = read(".agents/skills/custom-codereview-guide.md");

    expect(reviewGuide).not.toContain("Approving a same-repository PR runs");
    expect(reviewGuide).not.toContain("removes `.pr/` from the branch");
    expect(reviewGuide).toContain("cleanup PR");
  });
});
