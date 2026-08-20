import { beforeEach, describe, expect, it, vi } from "vitest";
import { RemoteWorkspace } from "@openhands/typescript-client/workspace/remote-workspace";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { uploadFilesToConversation } from "#/api/conversation-file-upload.api";
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

const batchGetCloudConversations = vi.fn();

vi.mock("#/api/cloud/conversation-service.api", () => ({
  batchGetCloudConversations: (...args: unknown[]) =>
    batchGetCloudConversations(...args),
}));

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Cloud",
  host: "https://app.all-hands.dev",
  apiKey: "cloud-token",
  kind: "cloud",
};

function makeFile(name: string) {
  return new File(["content"], name, { type: "text/plain" });
}

describe("uploadFilesToConversation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    __resetActiveStoreForTests();
    clearAgentServerHomeDirCache();
    fileUploadMock.mockResolvedValue(undefined);
    batchGetCloudConversations.mockReset();
    getHomeMock.mockReset();
    getHomeMock.mockResolvedValue({ home: "/Users/test" });
  });

  // @spec WUP-001 — Default-fallback relative working dirs are resolved
  // against /api/file/home, not the filesystem root.
  it("uploads local conversations under the agent-server home dir when working dir is relative", async () => {
    setRegisteredBackends([
      {
        id: "local-1",
        name: "Local",
        host: "http://127.0.0.1:18000",
        apiKey: "local-key",
        kind: "local",
      },
    ]);
    setActiveSelection({ backendId: "local-1" });

    const result = await uploadFilesToConversation("conv-1", [
      makeFile("a.txt"),
    ]);

    expect(fileUploadMock).toHaveBeenCalledWith(
      expect.objectContaining({ name: "a.txt" }),
      "/Users/test/workspace/project/a.txt",
    );
    expect(result.uploaded_files).toEqual(["a.txt"]);
    expect(batchGetCloudConversations).not.toHaveBeenCalled();
    expect(getHomeMock).toHaveBeenCalled();
  });

  // @spec WUP-001 — Absolute working dirs (e.g. the conversation's own
  // `workspace.working_dir`) pass through without a /api/file/home round-trip.
  it("respects an absolute conversation working_dir verbatim", async () => {
    setRegisteredBackends([
      {
        id: "local-1",
        name: "Local",
        host: "http://127.0.0.1:18000",
        apiKey: "local-key",
        kind: "local",
      },
    ]);
    setActiveSelection({ backendId: "local-1" });

    const result = await uploadFilesToConversation(
      "conv-1",
      [makeFile("notes.md")],
      {
        id: "conv-1",
        workspace: { working_dir: "/Users/test/projects/foo" },
      } as never,
    );

    expect(fileUploadMock).toHaveBeenCalledWith(
      expect.objectContaining({ name: "notes.md" }),
      "/Users/test/projects/foo/notes.md",
    );
    expect(result.uploaded_files).toEqual(["notes.md"]);
    expect(getHomeMock).not.toHaveBeenCalled();
  });

  it("uploads cloud conversations against the provisioned runtime URL", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    batchGetCloudConversations.mockResolvedValue([
      {
        id: "1717df59-63ee-43bf-b32a-83428d3efdc8",
        conversation_url:
          "http://runtime.example.dev/api/conversations/1717df59-63ee-43bf-b32a-83428d3efdc8",
        session_api_key: "runtime-session-key",
        workspace: { working_dir: "/workspace/project" },
      },
    ]);

    const conversationId = "1717df59-63ee-43bf-b32a-83428d3efdc8";
    const result = await uploadFilesToConversation(conversationId, [
      makeFile("notes.md"),
    ]);

    expect(batchGetCloudConversations).toHaveBeenCalledWith([conversationId]);
    expect(RemoteWorkspace).toHaveBeenCalledWith(
      expect.objectContaining({
        host: "http://runtime.example.dev",
        apiKey: "runtime-session-key",
      }),
    );
    expect(fileUploadMock).toHaveBeenCalledWith(
      expect.objectContaining({ name: "notes.md" }),
      "/workspace/project/notes.md",
    );
    expect(result.uploaded_files).toEqual(["notes.md"]);
  });
});
