import { RemoteWorkspace } from "@openhands/typescript-client/workspace/remote-workspace";
import { AxiosError } from "axios";
import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { callCloudProxy } from "#/api/cloud/proxy";
import type { Backend } from "#/api/backend-registry/types";
import { buildHttpBaseUrl } from "#/utils/websocket-url";
import AgentServerGitService from "../../src/api/git-service/agent-server-git-service.api";

const { mockGitChanges, mockGitDiff, mockClientGet } = vi.hoisted(() => ({
  mockGitChanges: vi.fn(),
  mockGitDiff: vi.fn(),
  mockClientGet: vi.fn(),
}));

vi.mock("@openhands/typescript-client/workspace/remote-workspace", () => ({
  RemoteWorkspace: vi.fn(function RemoteWorkspaceMock() {
    return {
      gitChanges: mockGitChanges,
      gitDiff: mockGitDiff,
      // The commit endpoints go through the SDK's raw HttpClient (the
      // typed wrappers don't know them).
      client: { get: mockClientGet },
    };
  }),
}));

vi.mock("#/api/cloud/proxy", () => ({
  callCloudProxy: vi.fn(),
}));

describe("AgentServerGitService", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGitChanges.mockReset();
    mockGitDiff.mockReset();
    mockClientGet.mockReset();
    vi.mocked(RemoteWorkspace).mockClear();
  });

  describe("getGitChanges", () => {
    test("throws when response is not an array (dead runtime returns HTML)", async () => {
      mockGitChanges.mockResolvedValue("<!DOCTYPE html><html>...</html>");

      await expect(
        AgentServerGitService.getGitChanges(
          "123",
          "http://localhost:3000/api/conversations/123",
          "test-api-key",
          "/workspace",
        ),
      ).rejects.toThrow("Invalid response from runtime");
    });

    test("passes conversation URL and session key to the runtime workspace", async () => {
      mockGitChanges.mockResolvedValue([]);

      await AgentServerGitService.getGitChanges(
        "123",
        "http://localhost:3000/api/conversations/123",
        "my-session-key",
        "/workspace/project",
      );

      expect(RemoteWorkspace).toHaveBeenCalledWith({
        host: "http://localhost:3000",
        apiKey: "my-session-key",
        workingDir: "workspace/project",
      });
      // No `ref`: the server auto-detects the base so committed (and
      // pushed) changes stay visible.
      expect(mockGitChanges).toHaveBeenCalledWith("/workspace/project");
    });

    test("preserves slashes in path when using workspace helper", async () => {
      mockGitChanges.mockResolvedValue([]);

      const pathWithSlashes = "/workspace/project/src/components";
      await AgentServerGitService.getGitChanges(
        "123",
        "http://localhost:3000/api/conversations/123",
        "test-api-key",
        pathWithSlashes,
      );

      expect(mockGitChanges).toHaveBeenCalledWith(pathWithSlashes);
    });

    test("maps V1 git statuses to V0 format", async () => {
      mockGitChanges.mockResolvedValue([
        { status: "ADDED", path: "new-file.ts" },
        { status: "DELETED", path: "removed-file.ts" },
        { status: "UPDATED", path: "changed-file.ts" },
        { status: "MOVED", path: "renamed-file.ts" },
      ]);

      const result = await AgentServerGitService.getGitChanges(
        "123",
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
    test("passes conversation URL and path to runtime workspace diff helper", async () => {
      mockGitDiff.mockResolvedValue({
        diff: "--- a/file.ts\n+++ b/file.ts\n...",
      });

      await AgentServerGitService.getGitChangeDiff(
        "123",
        "http://localhost:3000/api/conversations/123",
        "test-api-key",
        "/workspace/project/file.ts",
      );

      expect(RemoteWorkspace).toHaveBeenCalledWith({
        host: "http://localhost:3000",
        apiKey: "test-api-key",
        workingDir: "workspace/project",
      });
      // No `ref`: must match getGitChanges' auto-detected base.
      expect(mockGitDiff).toHaveBeenCalledWith("/workspace/project/file.ts");
    });

    test("preserves slashes in file path when using workspace helper", async () => {
      mockGitDiff.mockResolvedValue({ diff: "diff content" });

      const filePath = "/workspace/project/src/components/Button.tsx";
      await AgentServerGitService.getGitChangeDiff(
        "123",
        "http://localhost:3000/api/conversations/123",
        "test-api-key",
        filePath,
      );

      expect(mockGitDiff).toHaveBeenCalledWith(filePath);
    });

    test("returns the diff data from the response", async () => {
      const expectedDiff = {
        diff: "--- a/file.ts\n+++ b/file.ts\n@@ -1,3 +1,4 @@\n+new line",
      };
      mockGitDiff.mockResolvedValue(expectedDiff);

      const result = await AgentServerGitService.getGitChangeDiff(
        "123",
        "http://localhost:3000/api/conversations/123",
        "test-api-key",
        "/workspace/file.ts",
      );

      expect(result).toEqual({
        modified: "",
        original: "",
        diff: expectedDiff.diff,
      });
    });
  });

  describe("getGitCommits", () => {
    test("fetches recent commits via the raw client and maps to camelCase", async () => {
      // Arrange
      mockClientGet.mockResolvedValue({
        data: {
          commits: [
            {
              sha: "a".repeat(40),
              short_sha: "aaaaaaa",
              subject: "add logging",
              author: "Agent",
              timestamp: "2026-07-10T12:00:00+07:00",
            },
          ],
          has_more: true,
        },
      });

      // Act
      const page = await AgentServerGitService.getGitCommits(
        "http://localhost:3000/api/conversations/123",
        "test-api-key",
        "/workspace/project",
      );

      // Assert
      expect(mockClientGet).toHaveBeenCalledWith("/api/git/commits", {
        params: { path: "/workspace/project", limit: "50" },
      });
      expect(page).toEqual({
        commits: [
          {
            sha: "a".repeat(40),
            shortSha: "aaaaaaa",
            subject: "add logging",
            author: "Agent",
            timestamp: "2026-07-10T12:00:00+07:00",
          },
        ],
        hasMore: true,
      });
    });

    test("resolves null when the agent server predates the endpoint (404)", async () => {
      // Arrange — the SDK HttpClient throws a raw HttpError carrying status.
      mockClientGet.mockRejectedValue(
        Object.assign(new Error("Not Found"), {
          name: "HttpError",
          status: 404,
        }),
      );

      // Act
      const page = await AgentServerGitService.getGitCommits(
        "http://localhost:3000/api/conversations/123",
        "test-api-key",
        "/workspace/project",
      );

      // Assert
      expect(page).toBeNull();
    });
  });

  describe("getCommitChanges", () => {
    test("fetches a commit's files and maps statuses to the client format", async () => {
      // Arrange
      const sha = "b".repeat(40);
      mockClientGet.mockResolvedValue({
        data: [{ status: "DELETED", path: "doomed.txt" }],
      });

      // Act
      const changes = await AgentServerGitService.getCommitChanges(
        "http://localhost:3000/api/conversations/123",
        "test-api-key",
        "/workspace/project",
        sha,
      );

      // Assert
      expect(mockClientGet).toHaveBeenCalledWith(
        `/api/git/commits/${sha}/changes`,
        { params: { path: "/workspace/project" } },
      );
      expect(changes).toEqual([{ status: "D", path: "doomed.txt" }]);
    });
  });

  describe("getGitChangeDiff with commit", () => {
    test("fetches the per-commit diff via the raw client, not the wrapper", async () => {
      // Arrange
      const sha = "c".repeat(40);
      mockClientGet.mockResolvedValue({
        data: { modified: "", original: "contents" },
      });

      // Act
      const diff = await AgentServerGitService.getGitChangeDiff(
        "123",
        "http://localhost:3000/api/conversations/123",
        "test-api-key",
        "/workspace/project/doomed.txt",
        sha,
      );

      // Assert
      expect(mockClientGet).toHaveBeenCalledWith("/api/git/diff", {
        params: { path: "/workspace/project/doomed.txt", commit: sha },
      });
      expect(mockGitDiff).not.toHaveBeenCalled();
      expect(diff).toEqual({ modified: "", original: "contents" });
    });
  });

  describe("cloud backend", () => {
    const cloudBackend: Backend = {
      id: "cloud-1",
      name: "Production",
      host: "https://app.all-hands.dev",
      apiKey: "cloud-key",
      kind: "cloud",
    };

    const runtimeConversationUrl =
      "https://abc123.prod-runtime.all-hands.dev/api/conversations/conv-1";

    beforeEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id, orgId: "org-1" });
      vi.mocked(callCloudProxy).mockReset();
    });

    afterEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
    });

    describe("getGitChanges", () => {
      test("fetches changes via the cloud app-conversations git endpoint and maps statuses", async () => {
        // Arrange
        vi.mocked(callCloudProxy).mockResolvedValue([
          { status: "ADDED", path: "new-file.ts" },
          { status: "UPDATED", path: "changed-file.ts" },
        ]);

        // Act
        const result = await AgentServerGitService.getGitChanges(
          "conv-1",
          runtimeConversationUrl,
          "session-key",
          "workspace/project",
        );

        // Assert — addressed by conversation id on the cloud API itself
        // (no hostOverride / session-api-key runtime hop), with the
        // relative git path normalized to an absolute runtime path.
        expect(callCloudProxy).toHaveBeenCalledWith({
          backend: cloudBackend,
          method: "GET",
          path: "/api/v1/app-conversations/conv-1/git/changes?path=%2Fworkspace%2Fproject",
        });
        expect(result).toEqual([
          { status: "A", path: "new-file.ts" },
          { status: "M", path: "changed-file.ts" },
        ]);
      });

      test("throws when the cloud endpoint returns a non-array response", async () => {
        // Arrange — a dead runtime can surface as a non-JSON-array body.
        vi.mocked(callCloudProxy).mockResolvedValue(
          "<!DOCTYPE html><html>...</html>",
        );

        // Act + Assert
        await expect(
          AgentServerGitService.getGitChanges(
            "conv-1",
            runtimeConversationUrl,
            "session-key",
            "workspace/project",
          ),
        ).rejects.toThrow("Invalid response from runtime");
      });
    });

    describe("getGitChangeDiff", () => {
      test("fetches the diff via the cloud app-conversations git endpoint", async () => {
        // Arrange
        vi.mocked(callCloudProxy).mockResolvedValue({
          original: "old content",
          modified: "new content",
        });

        // Act
        const result = await AgentServerGitService.getGitChangeDiff(
          "conv-1",
          runtimeConversationUrl,
          "session-key",
          "/workspace/project/src/file.ts",
        );

        // Assert
        expect(callCloudProxy).toHaveBeenCalledWith({
          backend: cloudBackend,
          method: "GET",
          path: "/api/v1/app-conversations/conv-1/git/diff?path=%2Fworkspace%2Fproject%2Fsrc%2Ffile.ts",
        });
        expect(result).toEqual({
          original: "old content",
          modified: "new content",
        });
      });
    });

    describe("getGitCommits", () => {
      test("reaches the runtime through the generic cloud-proxy envelope", async () => {
        // Arrange
        vi.mocked(callCloudProxy).mockResolvedValue({
          commits: [],
          has_more: false,
        });

        // Act
        await AgentServerGitService.getGitCommits(
          runtimeConversationUrl,
          "session-key",
          "workspace/project",
        );

        // Assert — session-api-key envelope to the conversation's runtime
        // host, with the git path normalized to an absolute runtime path.
        expect(callCloudProxy).toHaveBeenCalledWith({
          backend: cloudBackend,
          method: "GET",
          hostOverride: buildHttpBaseUrl(runtimeConversationUrl),
          path: "/api/git/commits?path=%2Fworkspace%2Fproject&limit=50",
          authMode: "session-api-key",
          sessionApiKey: "session-key",
        });
      });

      test("resolves null when the runtime predates the endpoint (404)", async () => {
        // Arrange — the proxy hop surfaces upstream failures as AxiosErrors.
        vi.mocked(callCloudProxy).mockRejectedValue(
          Object.assign(new AxiosError("Not Found"), {
            response: { status: 404 },
          }),
        );

        // Act
        const page = await AgentServerGitService.getGitCommits(
          runtimeConversationUrl,
          "session-key",
          "workspace/project",
        );

        // Assert
        expect(page).toBeNull();
      });
    });

    test("does not touch the runtime workspace SDK on cloud backends", async () => {
      // Arrange
      vi.mocked(callCloudProxy).mockResolvedValue([]);

      // Act
      await AgentServerGitService.getGitChanges(
        "conv-1",
        runtimeConversationUrl,
        "session-key",
        "workspace/project",
      );

      // Assert — the conversation's runtime URL must no longer be dialed
      // from the browser; the cloud API makes the runtime hop server-side.
      expect(RemoteWorkspace).not.toHaveBeenCalled();
    });
  });
});
