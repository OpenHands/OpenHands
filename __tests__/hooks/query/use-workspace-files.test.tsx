import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AgentServerRuntimeService from "#/api/runtime-service/agent-server-runtime-service";
import { useWorkspaceFiles } from "#/hooks/query/use-workspace-files";

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
  selected_repository: null as string | null,
  workspace: { working_dir: "/workspace/project" },
};

beforeEach(() => {
  useActiveBackendMock.mockReset();
  useActiveConversationMock.mockReset();
  useRuntimeIsReadyMock.mockReset();
  executeCommandSpy.mockReset();

  useRuntimeIsReadyMock.mockReturnValue(true);
  useActiveConversationMock.mockReturnValue({ data: conversation });
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

  it("lists files via bash find", async () => {
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
    expect(executeCommandSpy).toHaveBeenCalledWith(
      conversation.conversation_url,
      conversation.session_api_key,
      expect.stringContaining("find ."),
      "/workspace/project",
      30,
    );
  });
});

describe("useWorkspaceFiles — cloud backend", () => {
  beforeEach(() => useActiveBackendMock.mockReturnValue(makeBackend("cloud")));

  it("lists files via bash find through the runtime hop", async () => {
    // Arrange — Cloud no longer falls back to git-changes-only listing;
    // Canvas /api/cloud-proxy restores executeCommand for sandboxes.
    executeCommandSpy.mockResolvedValue({
      exit_code: 0,
      stdout: "./README.md\n./src/main.py\n",
      stderr: "",
    });

    // Act
    const { result } = renderHook(() => useWorkspaceFiles(), {
      wrapper: makeWrapper(),
    });

    // Assert
    await waitFor(() =>
      expect(result.current.data).toEqual(["README.md", "src/main.py"]),
    );
    expect(executeCommandSpy).toHaveBeenCalledTimes(1);
  });

  it("absolute-izes getGitPath cwd when workspace.working_dir is absent", async () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        ...conversation,
        selected_repository: "org/panentheon",
        workspace: undefined,
      },
    });
    executeCommandSpy.mockResolvedValue({
      exit_code: 0,
      stdout: "./README.md\n",
      stderr: "",
    });

    const { result } = renderHook(() => useWorkspaceFiles(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.data).toEqual(["README.md"]));
    expect(executeCommandSpy).toHaveBeenCalledWith(
      conversation.conversation_url,
      conversation.session_api_key,
      expect.stringContaining("find ."),
      "/workspace/project/panentheon",
      30,
    );
  });
});
