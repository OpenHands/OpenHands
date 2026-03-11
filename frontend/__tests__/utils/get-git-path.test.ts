import { describe, it, expect } from "vitest";
import { getGitPath } from "#/utils/get-git-path";

describe("getGitPath", () => {
  it("should return /workspace/project when no repository is selected", () => {
    expect(getGitPath(null)).toBe("/workspace/project");
    expect(getGitPath(undefined)).toBe("/workspace/project");
  });

  it("should handle standard owner/repo format (GitHub)", () => {
    expect(getGitPath("OpenHands/OpenHands")).toBe("/workspace/project/OpenHands");
    expect(getGitPath("facebook/react")).toBe("/workspace/project/react");
  });

  it("should handle nested group paths (GitLab)", () => {
    expect(getGitPath("modernhealth/frontend-guild/pan")).toBe("/workspace/project/pan");
    expect(getGitPath("group/subgroup/repo")).toBe("/workspace/project/repo");
    expect(getGitPath("a/b/c/d/repo")).toBe("/workspace/project/repo");
  });

  it("should handle single segment paths", () => {
    expect(getGitPath("repo")).toBe("/workspace/project/repo");
  });

  it("should handle empty string", () => {
    expect(getGitPath("")).toBe("/workspace/project");
  });
});
