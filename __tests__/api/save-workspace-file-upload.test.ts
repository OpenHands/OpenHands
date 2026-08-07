import { beforeEach, describe, expect, it, vi } from "vitest";
import { saveWorkspaceFile } from "#/api/save-workspace-file";

const uploadTextMock = vi.fn();
const resolveAbsoluteMock = vi.fn();
const resolveWorkingDirMock = vi.fn();
const resolveRuntimeMock = vi.fn();
const getActiveBackendMock = vi.fn();

vi.mock("@openhands/typescript-client/workspace/remote-workspace", () => ({
  RemoteWorkspace: class {
    uploadText = uploadTextMock;
  },
}));

vi.mock("#/api/workspace-upload-path", async (importOriginal) => {
  const real =
    await importOriginal<typeof import("#/api/workspace-upload-path")>();
  return {
    ...real,
    resolveAbsoluteWorkspacePath: (...args: unknown[]) =>
      resolveAbsoluteMock(...args),
    resolveConversationUploadWorkingDir: (...args: unknown[]) =>
      resolveWorkingDirMock(...args),
  };
});

vi.mock("#/api/conversation-file-upload.api", () => ({
  resolveConversationRuntime: (...args: unknown[]) =>
    resolveRuntimeMock(...args),
}));

vi.mock("#/api/backend-registry/active-store", () => ({
  getActiveBackend: () => getActiveBackendMock(),
}));

vi.mock("#/api/agent-server-client-options", () => ({
  getAgentServerClientOptions: () => ({ host: "http://localhost" }),
}));

describe("saveWorkspaceFile", () => {
  beforeEach(() => {
    uploadTextMock.mockReset();
    resolveAbsoluteMock.mockReset();
    resolveWorkingDirMock.mockReset();
    resolveRuntimeMock.mockReset();
    getActiveBackendMock.mockReset();

    getActiveBackendMock.mockReturnValue({
      backend: { kind: "local" },
    });
    resolveWorkingDirMock.mockResolvedValue("workspace/project/abc");
    resolveRuntimeMock.mockResolvedValue({
      conversationUrl: null,
      sessionApiKey: null,
    });
    resolveAbsoluteMock.mockResolvedValue(
      "/home/openhands/workspace/project/abc",
    );
    uploadTextMock.mockResolvedValue({ success: true });
  });

  it("uploads text to the absolute workspace-relative destination", async () => {
    await saveWorkspaceFile({
      conversation: {
        id: "conv-1",
        conversation_url: null,
        session_api_key: null,
      } as never,
      relativePath: "debug_shuffle.py",
      content: "print('ok')\n",
    });

    expect(uploadTextMock).toHaveBeenCalledWith(
      "print('ok')\n",
      "/home/openhands/workspace/project/abc/debug_shuffle.py",
      "debug_shuffle.py",
    );
  });
});
