import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AgentServerGitService from "#/api/git-service/agent-server-git-service.api";
import { useUnifiedGetGitChanges } from "#/hooks/query/use-unified-get-git-changes";
import type { GitChange } from "#/api/open-hands.types";

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "conv-1" }),
  useOptionalConversationId: () => ({ conversationId: "conv-1" }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      id: "conv-1",
      conversation_url: "https://runtime.example.com/api/conversations/conv-1",
      session_api_key: "session-key",
      selected_repository: null,
      workspace: { working_dir: "/workspace/project" },
    },
  }),
}));

vi.mock("#/hooks/use-runtime-is-ready", () => ({
  useRuntimeIsReady: () => true,
}));

const getGitChangesSpy = vi.spyOn(AgentServerGitService, "getGitChanges");

// Wrap in a real QueryClient so refetch/refetchInterval-driven updates flow
// through the cache exactly as they do in the app.
function renderHookWithClient() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return renderHook(() => useUnifiedGetGitChanges(), {
    wrapper: ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
}

const changedFile: GitChange = {
  status: "M",
  path: "src/index.ts",
};

describe("useUnifiedGetGitChanges", () => {
  beforeEach(() => {
    getGitChangesSpy.mockReset();
    getGitChangesSpy.mockResolvedValue([changedFile]);
  });

  it("exposes the current git changes from the server", async () => {
    const { result } = renderHookWithClient();

    await waitFor(() => expect(result.current.data).toEqual([changedFile]));
    expect(result.current.isLoading).toBe(false);
    expect(result.current.isError).toBe(false);
  });

  it("clears stale rows when a refetch returns an empty diff (after commit)", async () => {
    // Regression for the "diff still shows after commit" bug: the hook must
    // reflect the freshest server state, not keep a stale accumulation of rows
    // from an earlier response. Simulate: first fetch sees two changed files,
    // then a refetch (e.g. triggered by the commit invalidate) returns the now
    // empty diff.
    getGitChangesSpy
      .mockResolvedValueOnce([changedFile, { status: "A", path: "new.ts" }])
      .mockResolvedValue([]);

    const { result } = renderHookWithClient();

    await waitFor(() => expect(result.current.data).toHaveLength(2));

    // Trigger a background refetch; the new empty diff must replace the rows.
    await result.current.refetch();
    await waitFor(() => expect(result.current.data).toHaveLength(0));
  });
});
