/**
 * Tests for the onProgress callback added to uploadFilesToConversation (#16430).
 * Run: `npx vitest run __tests__/api/conversation-file-upload-progress`
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { uploadFilesToConversation, type UploadProgressInfo } from "#/api/conversation-file-upload.api";

const fileUploadMock = vi.fn();

vi.mock("#/api/workspace-upload-path", () => ({
  buildWorkspaceUploadPath: vi.fn().mockResolvedValue("/workspace/uploads/file.txt"),
  getSafeUploadFileName: vi.fn((name: string) => name),
  resolveConversationUploadWorkingDir: vi.fn().mockResolvedValue("workspace/project"),
}));

vi.mock("#/api/backend-registry/active-store", () => ({
  getActiveBackend: vi.fn(() => ({ backend: { kind: "local" } })),
}));

vi.mock("#/api/agent-server-client-options", () => ({
  getAgentServerClientOptions: vi.fn(() => ({})),
}));

vi.mock("@openhands/typescript-client/workspace/remote-workspace", () => ({
  RemoteWorkspace: vi.fn(function RemoteWorkspaceMock() {
    return { fileUpload: fileUploadMock };
  }),
}));

describe("uploadFilesToConversation — onProgress callback (#16430)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fileUploadMock.mockResolvedValue(undefined);
  });

  it("onProgress fires with percentage=100 after uploading a single file", async () => {
    const progressCalls: UploadProgressInfo[] = [];
    const onProgress = vi.fn((info: UploadProgressInfo) => progressCalls.push(info));

    const file = new File(["content"], "test.txt", { type: "text/plain" });

    await uploadFilesToConversation("conv-123", [file], undefined, onProgress);

    expect(onProgress).toHaveBeenCalledTimes(1);
    expect(progressCalls[0]).toMatchObject({
      completed: 1,
      total: 1,
      percentage: 100,
    });
  });

  it("onProgress fires multiple times for multiple batches", async () => {
    const progressCalls: UploadProgressInfo[] = [];
    const onProgress = vi.fn((info: UploadProgressInfo) => progressCalls.push(info));

    const files = Array.from({ length: 6 }, (_, i) =>
      new File(["x"], `file${i}.txt`, { type: "text/plain" }),
    );

    await uploadFilesToConversation("conv-456", files, undefined, onProgress);

    // FILE_UPLOAD_CONCURRENCY=5, so 6 files = 2 batches
    expect(onProgress).toHaveBeenCalledTimes(2);
    expect(progressCalls[0].completed).toBe(5);
    expect(progressCalls[0].total).toBe(6);
    expect(progressCalls[0].percentage).toBe(83);
    expect(progressCalls[1].completed).toBe(6);
    expect(progressCalls[1].percentage).toBe(100);
  });

  it("works fine without onProgress callback (no error)", async () => {
    const file = new File(["content"], "no-progress.txt");
    await expect(
      uploadFilesToConversation("conv-789", [file], undefined),
    ).resolves.not.toThrow();
  });
});
