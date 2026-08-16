import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useUnifiedUploadFiles } from "#/hooks/mutation/use-unified-upload-files";
import type { FileUploadSuccessResponse } from "#/api/open-hands.types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { WORKSPACE_QUERY_KEYS } from "#/hooks/query/query-keys";

const uploadFilesToConversationMock = vi.fn();

vi.mock("#/api/conversation-file-upload.api", () => ({
  uploadFilesToConversation: (...args: unknown[]) =>
    uploadFilesToConversationMock(...args),
}));

function makeFile(name: string) {
  return new File(["content"], name, { type: "text/plain" });
}

function createWrapper(client: QueryClient) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={client}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    );
  };
}

describe("useUnifiedUploadFiles", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    vi.clearAllMocks();
  });

  it("invalidates file_changes and workspace-files for the conversation when a file is uploaded to an absolute path", async () => {
    const conversationId = "conv-1";
    const absolutePath = "/workspace/project/notes.md";

    uploadFilesToConversationMock.mockResolvedValue({
      uploaded_files: [absolutePath],
      skipped_files: [],
    } as FileUploadSuccessResponse);

    const fileChangesKey = WORKSPACE_QUERY_KEYS.fileChanges(
      conversationId,
      "https://conversation.example",
      "session-key",
      "/workspace/project",
    );
    const workspaceFilesKey = WORKSPACE_QUERY_KEYS.files(
      conversationId,
      "http://localhost:3000",
      "session-key",
      "/workspace/project",
    );
    queryClient.setQueryData(fileChangesKey, []);
    queryClient.setQueryData(workspaceFilesKey, ["existing.txt"]);
    const wrapper = createWrapper(queryClient);

    const { result } = renderHook(() => useUnifiedUploadFiles(), {
      wrapper,
    });

    const response = await result.current.mutateAsync({
      conversationId,
      files: [makeFile("notes.md")],
    });

    expect(response.uploaded_files).toEqual([absolutePath]);
    await waitFor(() => {
      expect(queryClient.getQueryState(fileChangesKey)?.isInvalidated).toBe(
        true,
      );
      expect(queryClient.getQueryState(workspaceFilesKey)?.isInvalidated).toBe(
        true,
      );
    });
  });

  it("does not invalidate either prefix when every file is skipped", async () => {
    const conversationId = "conv-2";

    uploadFilesToConversationMock.mockResolvedValue({
      uploaded_files: [],
      skipped_files: [{ name: "notes.md", reason: "Upload failed" }],
    } as FileUploadSuccessResponse);

    const fileChangesKey = WORKSPACE_QUERY_KEYS.fileChanges(
      conversationId,
      "https://conversation.example",
      "session-key",
      "/workspace/project",
    );
    const workspaceFilesKey = WORKSPACE_QUERY_KEYS.files(
      conversationId,
      "http://localhost:3000",
      "session-key",
      "/workspace/project",
    );
    queryClient.setQueryData(fileChangesKey, []);
    queryClient.setQueryData(workspaceFilesKey, ["existing.txt"]);
    const wrapper = createWrapper(queryClient);

    const { result } = renderHook(() => useUnifiedUploadFiles(), {
      wrapper,
    });

    await result.current.mutateAsync({
      conversationId,
      files: [makeFile("notes.md")],
    });

    expect(queryClient.getQueryState(fileChangesKey)?.isInvalidated).toBe(
      false,
    );
    expect(queryClient.getQueryState(workspaceFilesKey)?.isInvalidated).toBe(
      false,
    );
  });
});
