import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const {
  batchGetCloudConversationsMock,
  getAgentServerClientOptionsMock,
  remoteWorkspaceConstructorMock,
  fileUploadMock,
  buildWorkspaceUploadPathMock,
  getSafeUploadFileNameMock,
  resolveConversationUploadWorkingDirMock,
} = vi.hoisted(() => ({
  batchGetCloudConversationsMock: vi.fn(),
  getAgentServerClientOptionsMock: vi.fn(),
  remoteWorkspaceConstructorMock: vi.fn(),
  fileUploadMock: vi.fn(),
  buildWorkspaceUploadPathMock: vi.fn(),
  getSafeUploadFileNameMock: vi.fn(),
  resolveConversationUploadWorkingDirMock: vi.fn(),
}));

vi.mock("@openhands/typescript-client/workspace/remote-workspace", () => ({
  RemoteWorkspace: class {
    constructor(options: unknown) {
      remoteWorkspaceConstructorMock(options);
    }

    fileUpload = fileUploadMock;
  },
}));

vi.mock("#/api/agent-server-client-options", () => ({
  getAgentServerClientOptions: getAgentServerClientOptionsMock,
}));

vi.mock("#/api/cloud/conversation-service.api", () => ({
  batchGetCloudConversations: batchGetCloudConversationsMock,
}));

vi.mock("#/api/workspace-upload-path", () => ({
  buildWorkspaceUploadPath: buildWorkspaceUploadPathMock,
  getSafeUploadFileName: getSafeUploadFileNameMock,
  resolveConversationUploadWorkingDir: resolveConversationUploadWorkingDirMock,
}));

import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { uploadFilesToConversation } from "./conversation-file-upload.api";

const cloudBackend: Backend = {
  id: "cloud-backend",
  name: "OpenHands Cloud",
  host: "https://cloud.example.test",
  apiKey: "",
  kind: "cloud",
};

const cloudRuntime = {
  conversation_url: "https://runtime.example.com",
  session_api_key: "cloud-session-key",
};

const resolvedRuntime = {
  conversationUrl: cloudRuntime.conversation_url,
  sessionApiKey: cloudRuntime.session_api_key,
};

function cachedConversation(
  overrides: Partial<AppConversation> = {},
): AppConversation {
  return {
    id: "conv-1",
    title: "Cloud conversation",
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    trigger: null,
    pr_number: [],
    llm_model: null,
    metrics: null,
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    execution_status: "idle",
    conversation_url: "",
    session_api_key: "",
    sandbox_id: "sandbox-1",
    sub_conversation_ids: [],
    ...overrides,
  } as AppConversation;
}

function arrangeBackend(backend: Backend): void {
  setRegisteredBackends([backend]);
  setActiveSelection({ backendId: backend.id });
  getAgentServerClientOptionsMock.mockReturnValue({
    host: "https://runtime.example.com",
    apiKey: "cloud-session-key",
    workingDir: "workspace/project",
  });
  getSafeUploadFileNameMock.mockReturnValue("file.txt");
  buildWorkspaceUploadPathMock.mockResolvedValue("workspace/project/file.txt");
  resolveConversationUploadWorkingDirMock.mockResolvedValue(
    "workspace/project",
  );
}

describe("uploadFilesToConversation", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    fileUploadMock.mockResolvedValue(undefined);
  });

  afterEach(() => {
    setActiveSelection(null);
    setRegisteredBackends([]);
  });

  it("uses the fresh cloud runtime over cached empty-string fields", async () => {
    arrangeBackend(cloudBackend);
    batchGetCloudConversationsMock.mockResolvedValue([cloudRuntime]);
    const file = new File(["hello"], "file.txt");

    const result = await uploadFilesToConversation(
      "conv-1",
      [file],
      cachedConversation(),
    );

    expect(result).toEqual({ uploaded_files: ["file.txt"], skipped_files: [] });
    expect(batchGetCloudConversationsMock).toHaveBeenCalledWith(["conv-1"]);
    expect(getAgentServerClientOptionsMock).toHaveBeenCalledWith(
      expect.objectContaining({
        conversationUrl: resolvedRuntime.conversationUrl,
        sessionApiKey: resolvedRuntime.sessionApiKey,
      }),
    );
    expect(remoteWorkspaceConstructorMock).toHaveBeenCalledTimes(1);
    expect(fileUploadMock).toHaveBeenCalledWith(
      file,
      "workspace/project/file.txt",
    );
  });

  it("still surfaces the sandbox-starting error when no runtime is available", async () => {
    arrangeBackend(cloudBackend);
    batchGetCloudConversationsMock.mockResolvedValue([
      { conversation_url: null, session_api_key: null },
    ]);

    await expect(
      uploadFilesToConversation(
        "conv-1",
        [new File(["hello"], "file.txt")],
        cachedConversation(),
      ),
    ).rejects.toThrow("Conversation sandbox is still starting");
    expect(remoteWorkspaceConstructorMock).not.toHaveBeenCalled();
  });
});
