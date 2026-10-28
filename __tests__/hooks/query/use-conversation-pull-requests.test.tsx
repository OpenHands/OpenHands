import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { useConversationPullRequests } from "#/hooks/query/use-conversation-pull-requests";
import { GitProviderPullRequestStatusService } from "#/api/git-provider-pull-request-status-service";

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}

describe("useConversationPullRequests", () => {
  const getConversationPullRequestsSpy = vi.spyOn(
    GitProviderPullRequestStatusService,
    "getConversationPullRequests",
  );

  beforeEach(() => {
    getConversationPullRequestsSpy.mockReset();
  });

  it("loads the conversation's PR status from the provider", async () => {
    // Arrange
    const result = {
      status: "ok" as const,
      source: "tracked" as const,
      pullRequests: [],
      omittedCount: 0,
    };
    getConversationPullRequestsSpy.mockResolvedValue(result);

    // Act
    const { result: hook } = renderHook(
      () =>
        useConversationPullRequests({
          repository: "acme/widgets",
          provider: "github",
          branch: "fix-login",
          prNumbers: [7],
        }),
      { wrapper: makeWrapper() },
    );

    // Assert
    await waitFor(() => expect(hook.current.isSuccess).toBe(true));
    expect(getConversationPullRequestsSpy).toHaveBeenCalledWith({
      repository: "acme/widgets",
      provider: "github",
      branch: "fix-login",
      prNumbers: [7],
    });
    expect(hook.current.data).toEqual(result);
  });

  it("stays idle until the conversation has a repository and provider", () => {
    // Act
    const { result: hook } = renderHook(
      () =>
        useConversationPullRequests({
          repository: null,
          provider: "github",
          branch: "fix-login",
          prNumbers: [7],
        }),
      { wrapper: makeWrapper() },
    );

    // Assert
    expect(hook.current.fetchStatus).toBe("idle");
    expect(getConversationPullRequestsSpy).not.toHaveBeenCalled();
  });
});
