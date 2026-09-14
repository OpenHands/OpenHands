import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import AgentServerGitService from "#/api/git-service/agent-server-git-service.api";
import type { GitChange } from "#/api/open-hands.types";
import { useUnifiedGetGitChanges } from "#/hooks/query/use-unified-get-git-changes";

vi.mock("#/api/git-service/agent-server-git-service.api", () => ({
  default: {
    getGitChanges: vi.fn(),
  },
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "test-conversation-id" }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      conversation_url: null,
      session_api_key: null,
      selected_repository: null,
      workspace: { working_dir: "/workspace" },
    },
  }),
}));

vi.mock("#/hooks/use-runtime-is-ready", () => ({
  useRuntimeIsReady: () => true,
}));

const buildChanges = (paths: string[]): GitChange[] =>
  paths.map((path) => ({ path }) as GitChange);

const renderUseUnifiedGetGitChanges = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  const wrapper = ({ children }: React.PropsWithChildren) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  return renderHook(() => useUnifiedGetGitChanges(), { wrapper });
};

describe("useUnifiedGetGitChanges", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("does not return duplicate entries when the API response contains duplicate paths", async () => {
    vi.mocked(AgentServerGitService.getGitChanges).mockResolvedValue(
      buildChanges(["file-a.ts", "file-a.ts", "file-b.ts"]),
    );

    const { result } = renderUseUnifiedGetGitChanges();

    await waitFor(() => {
      expect(result.current.data.map((change) => change.path)).toEqual([
        "file-a.ts",
        "file-b.ts",
      ]);
    });
  });

  it("does not duplicate entries across refetches and keeps the latest changes on top", async () => {
    const getGitChangesMock = vi.mocked(AgentServerGitService.getGitChanges);
    getGitChangesMock.mockResolvedValueOnce(
      buildChanges(["file-a.ts", "file-b.ts"]),
    );

    const { result } = renderUseUnifiedGetGitChanges();

    await waitFor(() => {
      expect(result.current.data.map((change) => change.path)).toEqual([
        "file-a.ts",
        "file-b.ts",
      ]);
    });

    getGitChangesMock.mockResolvedValueOnce(
      buildChanges(["file-b.ts", "file-c.ts"]),
    );

    await result.current.refetch();

    await waitFor(() => {
      expect(result.current.data.map((change) => change.path)).toEqual([
        "file-c.ts",
        "file-b.ts",
      ]);
    });
  });
});
