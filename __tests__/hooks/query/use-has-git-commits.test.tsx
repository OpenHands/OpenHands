import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AgentServerRuntimeService from "#/api/runtime-service/agent-server-runtime-service";
import { useHasGitCommits } from "#/hooks/query/use-has-git-commits";

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
  return function HasGitCommitsTestWrapper({
    children,
  }: {
    children: React.ReactNode;
  }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };
}

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

const conversation = {
  id: "conv-1",
  conversation_url: "https://runtime.example.com/api/conversations/conv-1",
  session_api_key: "session-key",
  workspace: { working_dir: "/workspace/project" },
};

describe("useHasGitCommits", () => {
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

  it("does not call executeCommand on a cloud backend", async () => {
    // Arrange
    useActiveBackendMock.mockReturnValue(makeBackend("cloud"));

    // Act
    const { result } = renderHook(() => useHasGitCommits(), {
      wrapper: makeWrapper(),
    });

    // Assert: probe stays disabled; hasCommits remains null.
    await new Promise((resolve) => {
      setTimeout(resolve, 20);
    });
    expect(executeCommandSpy).not.toHaveBeenCalled();
    expect(result.current.hasCommits).toBeNull();
  });

  it("probes HEAD via executeCommand on a local backend and returns true when the repo has commits", async () => {
    // Arrange
    useActiveBackendMock.mockReturnValue(makeBackend("local"));
    executeCommandSpy.mockResolvedValue({
      exit_code: 0,
      stdout: "abcdef0\n",
      stderr: "",
    });

    // Act
    const { result } = renderHook(() => useHasGitCommits(), {
      wrapper: makeWrapper(),
    });

    // Assert
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(executeCommandSpy).toHaveBeenCalledWith(
      conversation.conversation_url,
      conversation.session_api_key,
      "git rev-parse --verify HEAD",
      "/workspace/project",
      10,
    );
    expect(result.current.hasCommits).toBe(true);
  });
});
