import { describe, test, expect, vi, beforeEach } from "vitest";
import axios from "axios";
import V1GitService from "../../src/api/git-service/v1-git-service.api";

vi.mock("axios");

describe("V1GitService", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("getGitChanges", () => {
    test("throws when response is not an array (dead runtime returns HTML)", async () => {
      const htmlResponse = "<!DOCTYPE html><html>...</html>";
      vi.mocked(axios.get).mockResolvedValue({ data: htmlResponse });

      await expect(
        V1GitService.getGitChanges(
          "http://localhost:3000/api/conversations/123",
          "test-api-key",
          "/workspace",
        ),
      ).rejects.toThrow("Invalid response from runtime");
    });

    test("uses query parameters instead of path segments for the path", async () => {
      vi.mocked(axios.get).mockResolvedValue({ data: [] });

      await V1GitService.getGitChanges(
        "http://localhost:3000/api/conversations/123",
        "test-api-key",
        "/workspace/project",
      );

      expect(axios.get).toHaveBeenCalledTimes(1);
      const [url, config] = vi.mocked(axios.get).mock.calls[0];

      // URL should NOT contain the path - it should end with /api/git/changes
      expect(url).toContain("/api/git/changes");
      expect(url).not.toContain("/workspace/project");
      expect(url).not.toContain(encodeURIComponent("/workspace/project"));

      // Path should be passed as a query parameter
      expect(config).toHaveProperty("params");
      expect(config?.params).toEqual({ path: "/workspace/project" });
    });

    test("preserves slashes in path when using query parameters", async () => {
      vi.mocked(axios.get).mockResolvedValue({ data: [] });

      const pathWithSlashes = "/workspace/project/src/components";
      await V1GitService.getGitChanges(
        "http://localhost:3000/api/conversations/123",
        "test-api-key",
        pathWithSlashes,
      );

      const [, config] = vi.mocked(axios.get).mock.calls[0];

      // Path should be preserved exactly as provided (slashes intact)
      expect(config?.params).toEqual({ path: pathWithSlashes });
    });

    test("includes session API key in headers when provided", async () => {
      vi.mocked(axios.get).mockResolvedValue({ data: [] });

      await V1GitService.getGitChanges(
        "http://localhost:3000/api/conversations/123",
        "my-session-key",
        "/workspace",
      );

      const [, config] = vi.mocked(axios.get).mock.calls[0];
      expect(config?.headers).toEqual({ "X-Session-API-Key": "my-session-key" });
    });

    test("maps V1 git statuses to V0 format", async () => {
      vi.mocked(axios.get).mockResolvedValue({
        data: [
          { status: "ADDED", path: "new-file.ts" },
          { status: "DELETED", path: "removed-file.ts" },
          { status: "UPDATED", path: "changed-file.ts" },
          { status: "MOVED", path: "renamed-file.ts" },
        ],
      });

      const result = await V1GitService.getGitChanges(
        "http://localhost:3000/api/conversations/123",
        "test-api-key",
        "/workspace",
      );

      expect(result).toEqual([
        { status: "A", path: "new-file.ts" },
        { status: "D", path: "removed-file.ts" },
        { status: "M", path: "changed-file.ts" },
        { status: "R", path: "renamed-file.ts" },
      ]);
    });
  });

  describe("getGitChangeDiff", () => {
    test("uses query parameters instead of path segments for the path", async () => {
      vi.mocked(axios.get).mockResolvedValue({
        data: { diff: "--- a/file.ts\n+++ b/file.ts\n..." },
      });

      await V1GitService.getGitChangeDiff(
        "http://localhost:3000/api/conversations/123",
        "test-api-key",
        "/workspace/project/file.ts",
      );

      expect(axios.get).toHaveBeenCalledTimes(1);
      const [url, config] = vi.mocked(axios.get).mock.calls[0];

      // URL should NOT contain the path - it should end with /api/git/diff
      expect(url).toContain("/api/git/diff");
      expect(url).not.toContain("/workspace/project/file.ts");
      expect(url).not.toContain(encodeURIComponent("/workspace/project/file.ts"));

      // Path should be passed as a query parameter
      expect(config).toHaveProperty("params");
      expect(config?.params).toEqual({ path: "/workspace/project/file.ts" });
    });

    test("preserves slashes in file path when using query parameters", async () => {
      vi.mocked(axios.get).mockResolvedValue({
        data: { diff: "diff content" },
      });

      const filePath = "/workspace/project/src/components/Button.tsx";
      await V1GitService.getGitChangeDiff(
        "http://localhost:3000/api/conversations/123",
        "test-api-key",
        filePath,
      );

      const [, config] = vi.mocked(axios.get).mock.calls[0];

      // Path should be preserved exactly as provided (slashes intact)
      expect(config?.params).toEqual({ path: filePath });
    });

    test("includes session API key in headers when provided", async () => {
      vi.mocked(axios.get).mockResolvedValue({
        data: { diff: "diff content" },
      });

      await V1GitService.getGitChangeDiff(
        "http://localhost:3000/api/conversations/123",
        "my-session-key",
        "/workspace/file.ts",
      );

      const [, config] = vi.mocked(axios.get).mock.calls[0];
      expect(config?.headers).toEqual({ "X-Session-API-Key": "my-session-key" });
    });

    test("returns the diff data from the response", async () => {
      const expectedDiff = {
        diff: "--- a/file.ts\n+++ b/file.ts\n@@ -1,3 +1,4 @@\n+new line",
      };
      vi.mocked(axios.get).mockResolvedValue({ data: expectedDiff });

      const result = await V1GitService.getGitChangeDiff(
        "http://localhost:3000/api/conversations/123",
        "test-api-key",
        "/workspace/file.ts",
      );

      expect(result).toEqual(expectedDiff);
    });
  });
});
