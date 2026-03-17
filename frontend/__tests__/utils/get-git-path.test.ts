import { describe, it, expect } from "vitest";
import { getGitPath } from "#/utils/get-git-path";

describe("getGitPath", () => {
  const conversationId = "abc123";

  it("should return {conversationId} when no repository is selected", () => {
    expect(getGitPath(conversationId, null)).toBe(conversationId);
    expect(getGitPath(conversationId, undefined)).toBe(conversationId);
  });

  it("should handle standard owner/repo format (GitHub)", () => {
    expect(getGitPath(conversationId, "OpenHands/OpenHands")).toBe(`${conversationId}/OpenHands`);
    expect(getGitPath(conversationId, "facebook/react")).toBe(`${conversationId}/react`);
  });

  it("should handle nested group paths (GitLab)", () => {
    expect(getGitPath(conversationId, "modernhealth/frontend-guild/pan")).toBe(`${conversationId}/pan`);
    expect(getGitPath(conversationId, "group/subgroup/repo")).toBe(`${conversationId}/repo`);
    expect(getGitPath(conversationId, "a/b/c/d/repo")).toBe(`${conversationId}/repo`);
  });

  it("should handle single segment paths", () => {
    expect(getGitPath(conversationId, "repo")).toBe(`${conversationId}/repo`);
  });

  it("should handle empty string", () => {
    expect(getGitPath(conversationId, "")).toBe(conversationId);
  });
});
