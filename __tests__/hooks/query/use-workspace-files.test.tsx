import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AgentServerRuntimeService from "#/api/runtime-service/agent-server-runtime-service";
import { useWorkspaceFiles } from "#/hooks/query/use-workspace-files";

const useActiveConversationMock = vi.fn();
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));

const useRuntimeIsReadyMock = vi.fn();
vi.mock("#/hooks/use-runtime-is-ready", () => ({
  useRuntimeIsReady: () => useRuntimeIsReadyMock(),
}));

const executeCommandSpy = vi.spyOn(AgentServerRuntimeService, "executeCommand");

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function WorkspaceFilesTestWrapper({
    children,
  }: {
    children: React.ReactNode;
  }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };
}

const conversation = {
  id: "conv-1",
  conversation_url: "https://runtime.example.com/api/conversations/conv-1",
  session_api_key: "session-key",
  workspace: { working_dir: "/workspace/project" },
};

beforeEach(() => {
  useActiveConversationMock.mockReset();
  useRuntimeIsReadyMock.mockReset();
  executeCommandSpy.mockReset();

  useRuntimeIsReadyMock.mockReturnValue(true);
  useActiveConversationMock.mockReturnValue({ data: conversation });
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("useWorkspaceFiles", () => {
  it("lists files via bash find for both local and cloud backends", async () => {
    executeCommandSpy.mockResolvedValue({
      exit_code: 0,
      stdout: "./hello.txt\n./src/index.ts\n",
      stderr: "",
    });

    const { result } = renderHook(() => useWorkspaceFiles(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() =>
      expect(result.current.data).toEqual(["hello.txt", "src/index.ts"]),
    );
    expect(executeCommandSpy).toHaveBeenCalledTimes(1);
  });

  it("normalizes paths by stripping leading ./", async () => {
    executeCommandSpy.mockResolvedValue({
      exit_code: 0,
      stdout: "./foo.txt\n./bar/baz.txt\n",
      stderr: "",
    });

    const { result } = renderHook(() => useWorkspaceFiles(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() =>
      expect(result.current.data).toEqual(["foo.txt", "bar/baz.txt"]),
    );
  });

  it("handles command failure gracefully", async () => {
    executeCommandSpy.mockResolvedValue({
      exit_code: 1,
      stdout: "",
      stderr: "some error",
    });

    const { result } = renderHook(() => useWorkspaceFiles(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
  });
});
