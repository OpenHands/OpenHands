import { describe, expect, it, vi, beforeEach } from "vitest";
import { RemoteWorkspace } from "@openhands/typescript-client/workspace/remote-workspace";

import ConversationService from "#/api/conversation-service/conversation-service.api";
import { clearAgentServerHomeDirCache } from "#/api/agent-server-home";

const fileUploadMock = vi.fn();
const getHomeMock = vi.fn();

vi.mock("@openhands/typescript-client/workspace/remote-workspace", () => ({
  RemoteWorkspace: vi.fn(function RemoteWorkspaceMock() {
    return { fileUpload: fileUploadMock };
  }),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  FileClient: vi.fn(function FileClientMock() {
    return { getHome: getHomeMock };
  }),
}));

function makeFile(name: string) {
  return new File(["content"], name, { type: "text/plain" });
}

describe("ConversationService", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    ConversationService.setCurrentConversation(null);
    clearAgentServerHomeDirCache();
    getHomeMock.mockResolvedValue({ home: "/Users/agent" });
  });

  describe("uploadFiles", () => {
    // @spec WUP-001 — The default fallback working dir is relative
    // (`workspace/project`); the upload path is resolved against the
    // agent-server's home directory via /api/file/home.
    it("uploads files through RemoteWorkspace and reports successes", async () => {
      fileUploadMock.mockResolvedValue(undefined);

      const result = await ConversationService.uploadFiles("conv-1", [
        makeFile("a.txt"),
        makeFile("b.txt"),
      ]);

      expect(RemoteWorkspace).toHaveBeenCalledWith(
        expect.objectContaining({
          host: expect.any(String),
          workingDir: expect.any(String),
        }),
      );
      expect(fileUploadMock).toHaveBeenCalledWith(
        expect.objectContaining({ name: "a.txt" }),
        "/Users/agent/workspace/project/a.txt",
      );
      expect(fileUploadMock).toHaveBeenCalledWith(
        expect.objectContaining({ name: "b.txt" }),
        "/Users/agent/workspace/project/b.txt",
      );
      expect(result).toEqual({
        uploaded_files: ["a.txt", "b.txt"],
        skipped_files: [],
      });
    });

    it("uploads using only the basename of user-provided file names", async () => {
      fileUploadMock.mockResolvedValue(undefined);

      const result = await ConversationService.uploadFiles("conv-1", [
        makeFile("../../evil.txt"),
      ]);

      expect(fileUploadMock).toHaveBeenCalledWith(
        expect.objectContaining({ name: "../../evil.txt" }),
        "/Users/agent/workspace/project/evil.txt",
      );
      expect(result).toEqual({
        uploaded_files: ["evil.txt"],
        skipped_files: [],
      });
    });

    it("uploads into the active conversation workspace when set", async () => {
      ConversationService.setCurrentConversation({
        id: "conv-1",
        workspace: { working_dir: "/workspace/project/my-app" },
      } as never);
      fileUploadMock.mockResolvedValue(undefined);

      await ConversationService.uploadFiles("conv-1", [makeFile("doc.txt")]);

      expect(fileUploadMock).toHaveBeenCalledWith(
        expect.objectContaining({ name: "doc.txt" }),
        "/workspace/project/my-app/doc.txt",
      );
    });

    it("uses the current conversation session key and reports per-file failures", async () => {
      ConversationService.setCurrentConversation({
        id: "conv-1",
        session_api_key: "session-key",
      } as never);
      fileUploadMock
        .mockResolvedValueOnce(undefined)
        .mockRejectedValueOnce(new Error("too large"));

      const result = await ConversationService.uploadFiles("conv-1", [
        makeFile("ok.txt"),
        makeFile("bad.txt"),
      ]);

      expect(RemoteWorkspace).toHaveBeenCalledWith(
        expect.objectContaining({ apiKey: "session-key" }),
      );
      expect(result).toEqual({
        uploaded_files: ["ok.txt"],
        skipped_files: [{ name: "bad.txt", reason: "too large" }],
      });
    });

    it("uploads files in bounded batches", async () => {
      let activeUploads = 0;
      let maxActiveUploads = 0;
      fileUploadMock.mockImplementation(async () => {
        activeUploads += 1;
        maxActiveUploads = Math.max(maxActiveUploads, activeUploads);
        await Promise.resolve();
        activeUploads -= 1;
      });

      const files = Array.from({ length: 7 }, (_, index) =>
        makeFile(`file-${String(index)}.txt`),
      );

      const result = await ConversationService.uploadFiles("conv-1", files);

      expect(fileUploadMock).toHaveBeenCalledTimes(7);
      expect(maxActiveUploads).toBe(5);
      expect(result.uploaded_files).toEqual(files.map((file) => file.name));
      expect(result.skipped_files).toEqual([]);
    });
  });
});
