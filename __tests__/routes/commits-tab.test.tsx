import React from "react";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { MemoryRouter } from "react-router";
import { http, HttpResponse } from "msw";

import GitCommits from "#/routes/commits-tab";
import AgentServerGitService from "#/api/git-service/agent-server-git-service.api";
import { ConversationOverviewGitSection } from "#/components/features/conversation/conversation-overview-git-section";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { useConversationStateStore } from "#/stores/conversation-state-store";
import { server } from "#/mocks/node";
import type { GitCommit } from "#/api/open-hands.types";

const conversation = {
  conversation_url: "http://localhost:18000/api/conversations/c1",
  session_api_key: "test-key",
  workspace: { working_dir: "/workspace/project" },
  execution_status: ExecutionStatus.RUNNING,
};

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "c1" }),
  useOptionalConversationId: () => ({ conversationId: "c1" }),
}));
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: conversation }),
}));
vi.mock("#/hooks/use-conversation-primary-repository", () => ({
  useConversationPrimaryRepository: () => ({
    repository: "acme/app",
    provider: "github",
    branch: "main",
    isConnected: true,
  }),
}));
vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({ data: {} }),
}));
vi.mock("#/hooks/query/use-repository-git-items", () => ({
  useRepositoryPullRequests: () => ({ data: [] }),
}));

const makeCommit = (overrides: Partial<GitCommit> = {}): GitCommit => ({
  sha: "a".repeat(40),
  shortSha: "aaaaaaa",
  subject: "add logging",
  author: "Agent",
  timestamp: "2026-07-10T12:00:00+07:00",
  ...overrides,
});

function renderGitViews(ui = <GitCommits />) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(ui, {
    wrapper: ({ children }) => (
      <MemoryRouter>
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      </MemoryRouter>
    ),
  });
  return client;
}

const gitPanels = (
  <>
    <GitCommits />
    <ConversationOverviewGitSection />
  </>
);

describe("Commits Tab", () => {
  const getGitCommitsSpy = vi.spyOn(AgentServerGitService, "getGitCommits");
  const getCommitChangesSpy = vi.spyOn(
    AgentServerGitService,
    "getCommitChanges",
  );
  const getGitChangesSpy = vi.spyOn(AgentServerGitService, "getGitChanges");

  beforeEach(() => {
    getGitCommitsSpy.mockReset();
    getCommitChangesSpy.mockReset();
    getGitChangesSpy.mockReset();
    getGitChangesSpy.mockResolvedValue([]);
    conversation.execution_status = ExecutionStatus.RUNNING;
    useConversationStateStore.getState().reset();
  });

  it("shows the waiting state while the runtime is inactive", () => {
    // Arrange
    conversation.execution_status = ExecutionStatus.ERROR;
    getGitCommitsSpy.mockResolvedValue({ commits: [], hasMore: false });

    // Act
    renderGitViews();

    // Assert
    expect(screen.getByTestId("commits-tab-status")).toBeInTheDocument();
  });

  it("shows a paused state in both panels without fetching history", () => {
    conversation.execution_status = ExecutionStatus.PAUSED;

    renderGitViews(gitPanels);

    expect(screen.getByRole("main")).toHaveTextContent("COMMON$PAUSED");
    expect(
      screen.getByTestId("conversation-overview-commits-count"),
    ).toHaveTextContent("COMMON$PAUSED");
    expect(screen.queryByTestId("commits-tab-status")).not.toBeInTheDocument();
    expect(getGitCommitsSpy).not.toHaveBeenCalled();
  });

  it("keeps the empty tab and Overview entry available on an unsupported server", async () => {
    getGitCommitsSpy.mockResolvedValue(null);

    renderGitViews(gitPanels);

    expect(
      await screen.findByText("DIFF_VIEWER$NO_COMMITS"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("conversation-overview-commits")).toBeEnabled();
    expect(
      screen.getByTestId("conversation-overview-commits-count"),
    ).toHaveTextContent("-");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("waits for a live pause to end before fetching the first page", async () => {
    const { setExecutionStatus } = useConversationStateStore.getState();
    setExecutionStatus("c1", ExecutionStatus.PAUSED);
    const commit = makeCommit();
    getGitCommitsSpy.mockResolvedValue({ commits: [commit], hasMore: false });

    renderGitViews(gitPanels);

    expect(screen.getByRole("main")).toHaveTextContent("COMMON$PAUSED");
    expect(getGitCommitsSpy).not.toHaveBeenCalled();
    act(() => setExecutionStatus("c1", ExecutionStatus.RUNNING));
    expect(await screen.findByText(commit.subject)).toBeInTheDocument();
    expect(
      screen.getByTestId("conversation-overview-commits-count"),
    ).toHaveTextContent("1");
  });

  it("recovers both panels when a user retries a failed commit request", async () => {
    const user = userEvent.setup();
    const endpoint = "http://localhost:18000/api/git/commits";
    const commit = makeCommit();
    server.use(
      http.get("http://localhost:18000/server_info", () =>
        HttpResponse.json({ version: "1.99.0" }),
      ),
      http.get(
        endpoint,
        () => HttpResponse.json({ detail: "read failed" }, { status: 500 }),
        { once: true },
      ),
      http.get(endpoint, () =>
        HttpResponse.json({
          commits: [{ ...commit, short_sha: commit.shortSha }],
          has_more: false,
        }),
      ),
    );
    renderGitViews(gitPanels);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "DIFF_VIEWER$COMMITS_UNAVAILABLE",
    );
    expect(
      screen.queryByText("DIFF_VIEWER$NO_COMMITS"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("conversation-overview-commits-count"),
    ).toHaveTextContent("COMMON$FAILED_TO_LOAD");
    await user.click(
      screen.getByRole("button", { name: "AUTOMATIONS$ERROR_RETRY" }),
    );

    expect(await screen.findByText(commit.subject)).toBeInTheDocument();
    expect(
      screen.getByTestId("conversation-overview-commits-count"),
    ).toHaveTextContent("1");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("keeps working-tree changes visible alongside a history read error", async () => {
    getGitCommitsSpy.mockRejectedValue(new Error("history unavailable"));
    getGitChangesSpy.mockResolvedValue([{ path: "src/a.ts", status: "M" }]);

    renderGitViews();

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "DIFF_VIEWER$COMMITS_UNAVAILABLE",
    );
    expect(screen.getByTestId("uncommitted-changes-row")).toBeInTheDocument();
  });

  it("retains the last commit page and its count after a failed refresh", async () => {
    const commit = makeCommit();
    getGitCommitsSpy
      .mockResolvedValueOnce({ commits: [commit], hasMore: true })
      .mockRejectedValue(new Error("refresh failed"));
    const client = renderGitViews(gitPanels);
    await screen.findByText(commit.subject);

    await act(() => client.invalidateQueries({ queryKey: ["git_commits"] }));

    expect(getGitCommitsSpy).toHaveBeenCalledTimes(2);
    expect(screen.getByText(commit.subject)).toBeInTheDocument();
    expect(
      screen.getByTestId("conversation-overview-commits-count"),
    ).toHaveTextContent("1+");
  });

  it("shows the empty state when the repository has no commits", async () => {
    // Arrange
    getGitCommitsSpy.mockResolvedValue({ commits: [], hasMore: false });

    // Act
    renderGitViews();

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
    renderGitViews();

    // Assert
    expect(await screen.findByText("add logging")).toBeInTheDocument();
    expect(screen.getByText("fix tests")).toBeInTheDocument();
    expect(screen.getByText("aaaaaaa")).toBeInTheDocument();
    expect(screen.getByTestId("uncommitted-changes-row")).toBeInTheDocument();
  });

  it("shows Uncommitted alone when there are working-tree changes but no commits", async () => {
    // Arrange
    getGitCommitsSpy.mockResolvedValue({ commits: [], hasMore: false });
    getGitChangesSpy.mockResolvedValue([{ path: "src/a.ts", status: "M" }]);

    // Act
    renderGitViews();

    // Assert
    expect(
      await screen.findByTestId("uncommitted-changes-row"),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("commit-row")).not.toBeInTheDocument();
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
    renderGitViews();
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
    renderGitViews();

    // Assert
    expect(
      await screen.findByTestId("commit-list-cap-notice"),
    ).toBeInTheDocument();
  });
});
