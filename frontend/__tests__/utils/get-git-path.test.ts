import { describe, it, expect } from "vitest";
import { getGitPath } from "#/utils/get-git-path";

describe("getGitPath", () => {
  const conversationId = "abc123";

  describe("without sandbox grouping (NO_GROUPING)", () => {
    it("should return /workspace/project when no repository is selected", () => {
      expect(getGitPath(conversationId, null, false)).toBe("/workspace/project");
      expect(getGitPath(conversationId, undefined, false)).toBe(
        "/workspace/project",
      );
    });

    it("should handle standard owner/repo format (GitHub)", () => {
      expect(getGitPath(conversationId, "OpenHands/OpenHands", false)).toBe(
        "/workspace/project/OpenHands",
      );
      expect(getGitPath(conversationId, "facebook/react", false)).toBe(
        "/workspace/project/react",
      );
    });

    it("should handle nested group paths (GitLab)", () => {
      expect(
        getGitPath(conversationId, "modernhealth/frontend-guild/pan", false),
      ).toBe("/workspace/project/pan");
      expect(getGitPath(conversationId, "group/subgroup/repo", false)).toBe(
        "/workspace/project/repo",
      );
      expect(getGitPath(conversationId, "a/b/c/d/repo", false)).toBe(
        "/workspace/project/repo",
      );
    });

    it("should handle single segment paths", () => {
      expect(getGitPath(conversationId, "repo", false)).toBe(
        "/workspace/project/repo",
      );
    });

    it("should handle empty string", () => {
      expect(getGitPath(conversationId, "", false)).toBe("/workspace/project");
    });
  });

  describe("with sandbox grouping enabled", () => {
    it("should return /workspace/project/{conversationId} when no repository is selected", () => {
      expect(getGitPath(conversationId, null, true)).toBe(
        `/workspace/project/${conversationId}`,
      );
      expect(getGitPath(conversationId, undefined, true)).toBe(
        `/workspace/project/${conversationId}`,
      );
    });

    it("should handle standard owner/repo format (GitHub)", () => {
      expect(getGitPath(conversationId, "OpenHands/OpenHands", true)).toBe(
        `/workspace/project/${conversationId}/OpenHands`,
      );
      expect(getGitPath(conversationId, "facebook/react", true)).toBe(
        `/workspace/project/${conversationId}/react`,
      );
    });

    it("should handle nested group paths (GitLab)", () => {
      expect(
        getGitPath(conversationId, "modernhealth/frontend-guild/pan", true),
      ).toBe(`/workspace/project/${conversationId}/pan`);
      expect(getGitPath(conversationId, "group/subgroup/repo", true)).toBe(
        `/workspace/project/${conversationId}/repo`,
      );
      expect(getGitPath(conversationId, "a/b/c/d/repo", true)).toBe(
        `/workspace/project/${conversationId}/repo`,
      );
    });

    it("should handle single segment paths", () => {
      expect(getGitPath(conversationId, "repo", true)).toBe(
        `/workspace/project/${conversationId}/repo`,
      );
    });

    it("should handle empty string", () => {
      expect(getGitPath(conversationId, "", true)).toBe(
        `/workspace/project/${conversationId}`,
      );
    });
  });

  describe("default behavior (useSandboxGrouping defaults to false)", () => {
    it("should default to no sandbox grouping", () => {
      expect(getGitPath(conversationId, null)).toBe("/workspace/project");
      expect(getGitPath(conversationId, "owner/repo")).toBe(
        "/workspace/project/repo",
      );
    });
  });
});
