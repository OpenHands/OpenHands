import { describe, it, expect } from "vitest";
import { getGitPath } from "#/utils/get-git-path";

describe("getGitPath", () => {
  const conversationId = "abc123";

  it("should return /workspace/project/{conversationId} when no repository is selected", () => {
    expect(getGitPath(conversationId, null)).toBe(`/workspace/project/${conversationId}`);
    expect(getGitPath(conversationId, undefined)).toBe(`/workspace/project/${conversationId}`);
  });

  it("should handle standard owner/repo format (GitHub)", () => {
    expect(getGitPath(conversationId, "OpenHands/OpenHands")).toBe(`/workspace/project/${conversationId}/OpenHands`);
    expect(getGitPath(conversationId, "facebook/react")).toBe(`/workspace/project/${conversationId}/react`);
  });

  it("should handle nested group paths (GitLab)", () => {
    expect(getGitPath(conversationId, "modernhealth/frontend-guild/pan")).toBe(`/workspace/project/${conversationId}/pan`);
    expect(getGitPath(conversationId, "group/subgroup/repo")).toBe(`/workspace/project/${conversationId}/repo`);
    expect(getGitPath(conversationId, "a/b/c/d/repo")).toBe(`/workspace/project/${conversationId}/repo`);
  });

  it("should handle single segment paths", () => {
    expect(getGitPath(conversationId, "repo")).toBe(`/workspace/project/${conversationId}/repo`);
  });

  it("should handle empty string", () => {
    expect(getGitPath(conversationId, "")).toBe(`/workspace/project/${conversationId}`);
  });
});
