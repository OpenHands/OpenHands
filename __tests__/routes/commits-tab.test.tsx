import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { MemoryRouter } from "react-router";

import GitCommits from "#/routes/commits-tab";
import AgentServerGitService from "#/api/git-service/agent-server-git-service.api";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";
import type { GitCommit } from "#/api/open-hands.types";

const conversation = {
  conversation_url: "http://localhost:18000/api/conversations/c1",
  session_api_key: "test-key",
  workspace: { working_dir: "/workspace/project" },
};

vi.mock("#/hooks/use-agent-state");
vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "c1" }),
}));
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: conversation }),
}));
vi.mock("#/hooks/use-runtime-is-ready", () => ({
  useRuntimeIsReady: () => true,
}));

const makeCommit = (overrides: Partial<GitCommit> = {}): GitCommit => ({
  sha: "a".repeat(40),
  shortSha: "aaaaaaa",
  subject: "add logging",
  author: "Agent",
  timestamp: "2026-07-10T12:00:00+07:00",
  ...overrides,
});

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <MemoryRouter>
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      {children}
    </QueryClientProvider>
  </MemoryRouter>
);

describe("Commits Tab", () => {
  const getGitCommitsSpy = vi.spyOn(AgentServerGitService, "getGitCommits");
  const getCommitChangesSpy = vi.spyOn(
    AgentServerGitService,
    "getCommitChanges",
  );

  beforeEach(() => {
    getGitCommitsSpy.mockReset();
    getCommitChangesSpy.mockReset();
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.RUNNING,
    });
  });

  it("shows the waiting state while the runtime is inactive", () => {
    // Arrange
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.ERROR,
    });
    getGitCommitsSpy.mockResolvedValue({ commits: [], hasMore: false });

    // Act
    render(<GitCommits />, { wrapper });

    // Assert
    expect(screen.getByTestId("commits-tab-status")).toBeInTheDocument();
  });

  it("shows the empty state when the repository has no commits", async () => {
    // Arrange
    getGitCommitsSpy.mockResolvedValue({ commits: [], hasMore: false });

    // Act
    render(<GitCommits />, { wrapper });

    // Assert
    expect(
      await screen.findByText("DIFF_VIEWER$NO_COMMITS"),
    ).toBeInTheDocument();
  });

  it("lists the commits returned by the server", async () => {
    // Arrange
    getGitCommitsSpy.mockResolvedValue({
      commits: [
        makeCommit({ sha: "a".repeat(40), subject: "add logging" }),
        makeCommit({
          sha: "b".repeat(40),
          shortSha: "bbbbbbb",
          subject: "fix tests",
        }),
      ],
      hasMore: false,
    });

    // Act
    render(<GitCommits />, { wrapper });

    // Assert
    expect(await screen.findByText("add logging")).toBeInTheDocument();
    expect(screen.getByText("fix tests")).toBeInTheDocument();
    expect(screen.getByText("aaaaaaa")).toBeInTheDocument();
  });

  it("expanding a commit fetches and lists the files it changed", async () => {
    // Arrange
    const user = userEvent.setup();
    const sha = "a".repeat(40);
    getGitCommitsSpy.mockResolvedValue({
      commits: [makeCommit({ sha })],
      hasMore: false,
    });
    getCommitChangesSpy.mockResolvedValue([
      { status: "D", path: "doomed.txt" },
    ]);
    render(<GitCommits />, { wrapper });
    await screen.findByText("add logging");

    // Act
    await user.click(screen.getByTestId("commit-row-toggle"));

    // Assert — the commit's file list is fetched by sha and rendered.
    await waitFor(() =>
      expect(getCommitChangesSpy).toHaveBeenCalledWith(
        conversation.conversation_url,
        conversation.session_api_key,
        "/workspace/project",
        sha,
      ),
    );
    expect(await screen.findByText("doomed.txt")).toBeInTheDocument();
  });

  it("shows the cap notice when the history was truncated", async () => {
    // Arrange
    getGitCommitsSpy.mockResolvedValue({
      commits: [makeCommit()],
      hasMore: true,
    });

    // Act
    render(<GitCommits />, { wrapper });

    // Assert
    expect(
      await screen.findByTestId("commit-list-cap-notice"),
    ).toBeInTheDocument();
  });
});
