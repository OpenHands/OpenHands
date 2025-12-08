import { describe, it, expect } from "vitest";
import {
  extractPRUrls,
  containsPRUrl,
  getFirstPRUrl,
} from "#/utils/parse-pr-url";

describe("parse-pr-url", () => {
  describe("extractPRUrls", () => {
    it("should extract GitHub PR URLs", () => {
      const text = "Check out this PR: https://github.com/owner/repo/pull/123";
      const urls = extractPRUrls(text);
      expect(urls).toEqual(["https://github.com/owner/repo/pull/123"]);
    });

    it("should extract GitLab MR URLs", () => {
      const text =
        "Merge request: https://gitlab.com/owner/repo/-/merge_requests/456";
      const urls = extractPRUrls(text);
      expect(urls).toEqual([
        "https://gitlab.com/owner/repo/-/merge_requests/456",
      ]);
    });

    it("should extract Bitbucket PR URLs", () => {
      const text =
        "PR link: https://bitbucket.org/owner/repo/pull-requests/789";
      const urls = extractPRUrls(text);
      expect(urls).toEqual([
        "https://bitbucket.org/owner/repo/pull-requests/789",
      ]);
    });

    it("should extract Azure DevOps PR URLs", () => {
      const text =
        "Azure PR: https://dev.azure.com/org/project/_git/repo/pullrequest/101";
      const urls = extractPRUrls(text);
      expect(urls).toEqual([
        "https://dev.azure.com/org/project/_git/repo/pullrequest/101",
      ]);
    });

    it("should extract multiple PR URLs", () => {
      const text = `
        GitHub: https://github.com/owner/repo/pull/123
        GitLab: https://gitlab.com/owner/repo/-/merge_requests/456
      `;
      const urls = extractPRUrls(text);
      expect(urls).toHaveLength(2);
      expect(urls).toContain("https://github.com/owner/repo/pull/123");
      expect(urls).toContain(
        "https://gitlab.com/owner/repo/-/merge_requests/456",
      );
    });

    it("should handle self-hosted GitLab URLs", () => {
      const text =
        "Self-hosted: https://gitlab.example.com/owner/repo/-/merge_requests/123";
      const urls = extractPRUrls(text);
      expect(urls).toEqual([
        "https://gitlab.example.com/owner/repo/-/merge_requests/123",
      ]);
    });

    it("should return empty array when no PR URLs found", () => {
      const text = "This is just regular text with no PR URLs";
      const urls = extractPRUrls(text);
      expect(urls).toEqual([]);
    });

    it("should handle URLs with HTTP instead of HTTPS", () => {
      const text = "HTTP PR: http://github.com/owner/repo/pull/123";
      const urls = extractPRUrls(text);
      expect(urls).toEqual(["http://github.com/owner/repo/pull/123"]);
    });

    it("should remove duplicate URLs", () => {
      const text = `
        Same PR mentioned twice:
        https://github.com/owner/repo/pull/123
        https://github.com/owner/repo/pull/123
      `;
      const urls = extractPRUrls(text);
      expect(urls).toEqual(["https://github.com/owner/repo/pull/123"]);
    });
  });

  describe("containsPRUrl", () => {
    it("should return true when PR URL is present", () => {
      const text = "Check out this PR: https://github.com/owner/repo/pull/123";
      expect(containsPRUrl(text)).toBe(true);
    });

    it("should return false when no PR URL is present", () => {
      const text = "This is just regular text";
      expect(containsPRUrl(text)).toBe(false);
    });
  });

  describe("getFirstPRUrl", () => {
    it("should return the first PR URL found", () => {
      const text = `
        First: https://github.com/owner/repo/pull/123
        Second: https://gitlab.com/owner/repo/-/merge_requests/456
      `;
      const url = getFirstPRUrl(text);
      expect(url).toBe("https://github.com/owner/repo/pull/123");
    });

    it("should return null when no PR URL is found", () => {
      const text = "This is just regular text";
      const url = getFirstPRUrl(text);
      expect(url).toBeNull();
    });
  });

  describe("real-world scenarios", () => {
    it("should handle typical microagent finish messages", () => {
      const text = `
        I have successfully created a pull request with the requested changes.
        You can view the PR here: https://github.com/OpenHands/OpenHands/pull/1234

        The changes include:
        - Updated the component
        - Added tests
        - Fixed the issue
      `;
      const url = getFirstPRUrl(text);
      expect(url).toBe("https://github.com/OpenHands/OpenHands/pull/1234");
    });

    it("should handle messages with PR URLs in the middle", () => {
      const text = `
        Task completed successfully! I've created a pull request at
        https://github.com/owner/repo/pull/567 with all the requested changes.
        Please review when you have a chance.
      `;
      const url = getFirstPRUrl(text);
      expect(url).toBe("https://github.com/owner/repo/pull/567");
    });
  });
});
