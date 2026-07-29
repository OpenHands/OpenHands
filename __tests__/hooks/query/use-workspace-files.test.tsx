import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AgentServerRuntimeService from "#/api/runtime-service/agent-server-runtime-service";
import { useWorkspaceFiles } from "#/hooks/query/use-workspace-files";
import type { GitChange } from "#/api/open-hands.types";

const useActiveBackendMock = vi.fn();
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => useActiveBackendMock(),
}));

const useActiveConversationMock = vi.fn();
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));

const useRuntimeIsReadyMock = vi.fn();
vi.mock("#/hooks/use-runtime-is-ready", () => ({
  useRuntimeIsReady: () => useRuntimeIsReadyMock(),
}));

const useUnifiedGetGitChangesMock = vi.fn();
vi.mock("#/hooks/query/use-unified-get-git-changes", () => ({
  useUnifiedGetGitChanges: () => useUnifiedGetGitChangesMock(),
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

function gitChangesResult(data: GitChange[], isLoading = false) {
  return {
    data,
    isLoading,
    isFetching: false,
    isSuccess: true,
    isError: false,
    error: null,
    refetch: vi.fn(),
  };
}

beforeEach(() => {
  useActiveBackendMock.mockReset();
  useActiveConversationMock.mockReset();
  useRuntimeIsReadyMock.mockReset();
  useUnifiedGetGitChangesMock.mockReset();
  executeCommandSpy.mockReset();

  useRuntimeIsReadyMock.mockReturnValue(true);
  useActiveConversationMock.mockReturnValue({ data: conversation });
  useUnifiedGetGitChangesMock.mockReturnValue(gitChangesResult([]));
});

afterEach(() => {
  vi.clearAllMocks();
});

const makeBackend = (kind: "local" | "cloud") => ({
  backend: {
    id: "backend-id",
    name: kind === "local" ? "Local" : "Production",
    host:
      kind === "local" ? "http://127.0.0.1:8000" : "https://app.all-hands.dev",
    apiKey: "test-key",
    kind,
  },
  orgId: null,
});

describe("useWorkspaceFiles — local backend", () => {
  beforeEach(() => useActiveBackendMock.mockReturnValue(makeBackend("local")));

  it("lists files via bash find and does not touch git changes", async () => {
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
});

describe("useWorkspaceFiles — cloud backend", () => {
  beforeEach(() => useActiveBackendMock.mockReturnValue(makeBackend("cloud")));

  it("derives the file list from git changes without running bash", async () => {
    useUnifiedGetGitChangesMock.mockReturnValue(
      gitChangesResult([
        { status: "A", path: "hello.txt" },
        { status: "M", path: "src/index.ts" },
      ]),
    );

    const { result } = renderHook(() => useWorkspaceFiles(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() =>
      expect(result.current.data).toEqual(["hello.txt", "src/index.ts"]),
    );
    // Cloud must never drive the removed bash/cloud-proxy path.
    expect(executeCommandSpy).not.toHaveBeenCalled();
  });

  it("drops deleted files (they can't be opened) and de-dupes", async () => {
    useUnifiedGetGitChangesMock.mockReturnValue(
      gitChangesResult([
        { status: "A", path: "hello.txt" },
        { status: "D", path: "gone.txt" },
        { status: "M", path: "hello.txt" },
      ]),
    );

    const { result } = renderHook(() => useWorkspaceFiles(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.data).toEqual(["hello.txt"]));
  });

  it("surfaces the git-changes loading state", async () => {
    useUnifiedGetGitChangesMock.mockReturnValue(gitChangesResult([], true));

    const { result } = renderHook(() => useWorkspaceFiles(), {
      wrapper: makeWrapper(),
    });

    expect(result.current.isLoading).toBe(true);
    expect(executeCommandSpy).not.toHaveBeenCalled();
  });
});
