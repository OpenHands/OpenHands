import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { useUnifiedGitCommits } from "#/hooks/query/use-unified-git-commits";
import AgentServerGitService from "#/api/git-service/agent-server-git-service.api";

const conversation = {
  conversation_url: "http://localhost:18000/api/conversations/c1",
  session_api_key: "test-key",
  workspace: { working_dir: "/workspace/project" },
};

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "c1" }),
}));
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: conversation }),
}));
vi.mock("#/hooks/use-runtime-is-ready", () => ({
  useRuntimeIsReady: () => true,
}));

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}

describe("useUnifiedGitCommits", () => {
  const getGitCommitsSpy = vi.spyOn(AgentServerGitService, "getGitCommits");

  beforeEach(() => {
    getGitCommitsSpy.mockReset();
  });

  it("exposes the service's commit page for the active conversation", async () => {
    // Arrange
    const commit = {
      sha: "a".repeat(40),
      shortSha: "aaaaaaa",
      subject: "add logging",
      author: "Agent",
      timestamp: "2026-07-10T12:00:00+07:00",
    };
    getGitCommitsSpy.mockResolvedValue({ commits: [commit], hasMore: true });

    // Act
    const { result } = renderHook(() => useUnifiedGitCommits(), {
      wrapper: makeWrapper(),
    });

    // Assert
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(getGitCommitsSpy).toHaveBeenCalledWith(
      conversation.conversation_url,
      conversation.session_api_key,
      "/workspace/project",
      50,
    );
    expect(result.current.commits).toEqual([commit]);
    expect(result.current.hasMore).toBe(true);
    expect(result.current.isUnsupported).toBe(false);
  });

  it("marks the feature unsupported when the service resolves null (old server)", async () => {
    // Arrange
    getGitCommitsSpy.mockResolvedValue(null);

    // Act
    const { result } = renderHook(() => useUnifiedGitCommits(), {
      wrapper: makeWrapper(),
    });

    // Assert
    await waitFor(() => expect(result.current.isUnsupported).toBe(true));
    expect(result.current.commits).toEqual([]);
  });
});
