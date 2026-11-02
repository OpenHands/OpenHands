import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { useActiveConversationPullRequests } from "#/hooks/use-active-conversation-pull-requests";
import { GitProviderPullRequestStatusService } from "#/api/git-provider-pull-request-status-service";

const localGitInfo = vi.hoisted(() => ({
  current: undefined as
    | { repository: string | null; branch: string | null }
    | undefined,
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: { id: "c1", pr_number: [] } }),
}));
vi.mock("#/hooks/query/use-local-git-info", () => ({
  useLocalGitInfo: () => ({ data: localGitInfo.current }),
}));
vi.mock("#/hooks/use-conversation-primary-repository", () => ({
  useConversationPrimaryRepository: () => ({
    repository: "acme/widgets",
    provider: "github",
    branch: "main",
    isConnected: true,
  }),
}));

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}

describe("useActiveConversationPullRequests", () => {
  const getConversationPullRequestsSpy = vi.spyOn(
    GitProviderPullRequestStatusService,
    "getConversationPullRequests",
  );

  beforeEach(() => {
    getConversationPullRequestsSpy.mockReset();
    getConversationPullRequestsSpy.mockResolvedValue({
      status: "ok",
      source: "none",
      omittedCount: 0,
      pullRequests: [],
    });
  });

  it.each([
    // The agent's working branch beats the branch picked at start.
    [{ repository: "acme/widgets", branch: "fix-login" }, "fix-login"],
    // A detached HEAD or another repository's checkout says nothing.
    [{ repository: "acme/widgets", branch: "HEAD" }, "main"],
    [{ repository: "acme/other", branch: "fix-login" }, "main"],
  ])("matches PRs by the right branch (%o)", async (checkout, expected) => {
    // Arrange
    localGitInfo.current = checkout;

    // Act
    renderHook(() => useActiveConversationPullRequests(), {
      wrapper: makeWrapper(),
    });

    // Assert
    await waitFor(() =>
      expect(getConversationPullRequestsSpy).toHaveBeenCalledWith(
        expect.objectContaining({ branch: expected }),
      ),
    );
  });
});
