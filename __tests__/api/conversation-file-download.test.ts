import { beforeEach, describe, expect, it, vi } from "vitest";
import { FileClient } from "@openhands/typescript-client/clients";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { clearAgentServerHomeDirCache } from "#/api/agent-server-home";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { downloadConversationFile } from "#/api/conversation-file-download.api";

const downloadFile = vi.fn();
const close = vi.fn();
vi.mock("@openhands/typescript-client/clients", () => ({
  FileClient: vi.fn(function () {
    return {
      downloadFile,
      close,
      getHome: async () => ({ home: "/home/test" }),
    };
  }),
}));
const batchGetCloudConversations = vi.fn();
vi.mock("#/api/cloud/conversation-service.api", () => ({
  batchGetCloudConversations: (...args: unknown[]) =>
    batchGetCloudConversations(...args),
}));

function conversation(workingDir: string): AppConversation {
  return {
    id: "conv-1",
    workspace: { working_dir: workingDir },
  } as AppConversation;
}

describe("downloadConversationFile", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    __resetActiveStoreForTests();
    clearAgentServerHomeDirCache();
    setRegisteredBackends([
      {
        id: "local",
        name: "Local",
        kind: "local",
        host: "http://localhost:18000",
        apiKey: "test-key",
      },
      {
        id: "cloud",
        name: "Cloud",
        kind: "cloud",
        host: "https://cloud.example.com",
        apiKey: "test-cloud-key",
      },
    ]);
    setActiveSelection({ backendId: "local" });
    downloadFile.mockResolvedValue(new Uint8Array([0, 255, 128, 10]).buffer);
  });

  // @spec FD-001 — Preserve original bytes and resolve the conversation workspace
  it.each([
    ["/workspace/project", "/workspace/project/nested/café data.bin"],
    ["/", "/nested/café data.bin"],
    ["workspace/project", "/home/test/workspace/project/nested/café data.bin"],
    ["C:\\project", "C:\\project/nested/café data.bin"],
  ])("downloads bytes from %s", async (workingDir, expectedPath) => {
    const blob = await downloadConversationFile(
      conversation(workingDir),
      "nested/café data.bin",
    );

    expect(downloadFile).toHaveBeenCalledWith(expectedPath);
    expect(FileClient).toHaveBeenLastCalledWith(
      expect.objectContaining({
        conversationId: "conv-1",
        host: "http://localhost:18000",
        apiKey: "test-key",
      }),
    );
    const bytes = await new Promise<ArrayBuffer>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as ArrayBuffer);
      reader.readAsArrayBuffer(blob);
    });
    expect([...new Uint8Array(bytes)]).toEqual([0, 255, 128, 10]);
    expect(close).toHaveBeenCalledOnce();
  });

  // @spec FD-001 — Cloud downloads target the provisioned conversation runtime
  it("uses the Cloud runtime rather than the selected backend host", async () => {
    setActiveSelection({ backendId: "cloud" });
    batchGetCloudConversations.mockResolvedValue([
      {
        conversation_url:
          "https://runtime.example.com/api/conversations/conv-1",
        session_api_key: "test-runtime-key",
      },
    ]);

    await downloadConversationFile(
      conversation("/workspace/project"),
      "image.png",
    );

    expect(FileClient).toHaveBeenCalledWith(
      expect.objectContaining({
        host: "https://runtime.example.com",
        apiKey: "test-runtime-key",
        conversationId: "conv-1",
      }),
    );
  });

  it("does not fall back to local files when the Cloud runtime is unavailable", async () => {
    setActiveSelection({ backendId: "cloud" });
    batchGetCloudConversations.mockResolvedValue([null]);

    await expect(
      downloadConversationFile(conversation("/workspace/project"), "a.txt"),
    ).rejects.toThrow();

    expect(downloadFile).not.toHaveBeenCalled();
  });

  it("keeps the original backend when the active backend changes during a download", async () => {
    const pending = downloadConversationFile(
      conversation("/workspace/project"),
      "a.txt",
    );
    setActiveSelection({ backendId: "cloud" });

    await pending;

    expect(FileClient).toHaveBeenCalledWith(
      expect.objectContaining({
        host: "http://localhost:18000",
        apiKey: "test-key",
        conversationId: "conv-1",
      }),
    );
  });

  it("closes the client and propagates download failures", async () => {
    downloadFile.mockRejectedValueOnce(new Error("File not found"));

    await expect(
      downloadConversationFile(
        conversation("/workspace/project"),
        "missing.txt",
      ),
    ).rejects.toThrow("File not found");

    expect(close).toHaveBeenCalledOnce();
  });
});
