import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "test-utils";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { NewConversationButton } from "#/components/features/conversation-panel/new-conversation-button";
import { GitRepository } from "#/types/git";

const mockUseUserProviders = vi.fn();
const mockUseActiveBackend = vi.fn();
const mockUseGitRepositories = vi.fn();
const mockUseSearchRepositories = vi.fn();

vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: () => mockUseUserProviders(),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => mockUseActiveBackend(),
}));

vi.mock("#/hooks/query/use-git-repositories", () => ({
  useGitRepositories: (...args: unknown[]) => mockUseGitRepositories(...args),
}));

vi.mock("#/hooks/query/use-search-repositories", () => ({
  useSearchRepositories: (...args: unknown[]) =>
    mockUseSearchRepositories(...args),
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackConversationCreated: vi.fn(),
  }),
}));

const makeRepo = (
  full_name: string,
  overrides: Partial<GitRepository> = {},
): GitRepository => ({
  id: full_name,
  full_name,
  git_provider: "github",
  is_public: true,
  ...overrides,
});

const makeStartTask = (conversationId: string) => ({
  id: "task-id",
  created_by_user_id: null,
  status: "READY" as const,
  detail: null,
  app_conversation_id: conversationId,
  agent_server_url: "http://agent-server.local",
  request: {
    initial_message: null,
    processors: [],
    llm_model: null,
    selected_repository: null,
    selected_branch: null,
    git_provider: "github" as const,
    suggested_task: null,
    title: null,
    trigger: null,
    pr_number: [],
    parent_conversation_id: null,
    agent_type: "default" as const,
  },
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
});

describe("NewConversationButton (cloud)", () => {
  beforeEach(() => {
    window.localStorage.clear();
    mockUseActiveBackend.mockReturnValue({
      backend: { id: "cloud-1", kind: "cloud" },
      orgId: null,
    });
    mockUseUserProviders.mockReturnValue({
      providers: ["github"],
      isLoadingSettings: false,
    });
    mockUseGitRepositories.mockReturnValue({
      data: {
        pages: [
          {
            items: [
              makeRepo("octo/cat", { main_branch: "main" }),
              makeRepo("octo/dog", { main_branch: "trunk" }),
            ],
            next_page_id: null,
          },
        ],
      },
      isLoading: false,
      isError: false,
      hasNextPage: false,
      isFetchingNextPage: false,
      fetchNextPage: vi.fn(),
      onLoadMore: vi.fn(),
    });
    mockUseSearchRepositories.mockReturnValue({
      data: [],
      isLoading: false,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
  });

  it("renders a list of repositories from the active provider", async () => {
    const user = userEvent.setup();
    renderWithProviders(<NewConversationButton />);

    await user.click(screen.getByTestId("new-conversation-button"));

    expect(screen.getByTestId("new-conversation-popover")).toBeInTheDocument();
    const items = screen.getAllByTestId("launch-repository");
    expect(items).toHaveLength(2);
    expect(items[0]).toHaveAttribute("data-repo-name", "octo/cat");
    expect(items[1]).toHaveAttribute("data-repo-name", "octo/dog");

    // No workspace footer in cloud mode.
    expect(
      screen.queryByTestId("add-workspaces-button"),
    ).not.toBeInTheDocument();
    expect(screen.queryByTestId("launch-no-workspace")).not.toBeInTheDocument();
  });

  it("launches a conversation against the repo's default branch", async () => {
    const navigate = vi.fn();
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(makeStartTask("conv-xyz"));

    const user = userEvent.setup();
    renderWithProviders(<NewConversationButton />, {
      navigation: { navigate, currentPath: "/conversations" },
    });

    await user.click(screen.getByTestId("new-conversation-button"));
    const items = screen.getAllByTestId("launch-repository");
    await user.click(items[1]); // octo/dog, main_branch = "trunk"

    await waitFor(() => {
      expect(createSpy).toHaveBeenCalledWith(
        undefined,
        undefined,
        undefined,
        {
          selected_repository: "octo/dog",
          selected_branch: "trunk",
          git_provider: "github",
        },
        undefined,
        undefined,
        undefined,
        undefined,
      );
    });
    await waitFor(() => {
      expect(navigate).toHaveBeenCalledWith("/conversations/conv-xyz");
    });
  });

  it('falls back to "main" when the repository has no main_branch metadata', async () => {
    mockUseGitRepositories.mockReturnValue({
      data: {
        pages: [{ items: [makeRepo("octo/no-main")], next_page_id: null }],
      },
      isLoading: false,
      isError: false,
      hasNextPage: false,
      isFetchingNextPage: false,
      fetchNextPage: vi.fn(),
      onLoadMore: vi.fn(),
    });
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(makeStartTask("conv-abc"));

    const user = userEvent.setup();
    renderWithProviders(<NewConversationButton />);

    await user.click(screen.getByTestId("new-conversation-button"));
    await user.click(screen.getByTestId("launch-repository"));

    await waitFor(() => {
      expect(createSpy).toHaveBeenCalledWith(
        undefined,
        undefined,
        undefined,
        {
          selected_repository: "octo/no-main",
          selected_branch: "main",
          git_provider: "github",
        },
        undefined,
        undefined,
        undefined,
        undefined,
      );
    });
  });

  it("renders provider tabs when multiple providers are connected", async () => {
    mockUseUserProviders.mockReturnValue({
      providers: ["github", "gitlab"],
      isLoadingSettings: false,
    });

    const user = userEvent.setup();
    renderWithProviders(<NewConversationButton />);

    await user.click(screen.getByTestId("new-conversation-button"));

    expect(screen.getByTestId("cloud-provider-tabs")).toBeInTheDocument();
    expect(screen.getByTestId("cloud-provider-tab-github")).toBeInTheDocument();
    expect(screen.getByTestId("cloud-provider-tab-gitlab")).toBeInTheDocument();
  });

  it("shows an empty state when no repositories are returned", async () => {
    mockUseGitRepositories.mockReturnValue({
      data: { pages: [{ items: [], next_page_id: null }] },
      isLoading: false,
      isError: false,
      hasNextPage: false,
      isFetchingNextPage: false,
      fetchNextPage: vi.fn(),
      onLoadMore: vi.fn(),
    });

    const user = userEvent.setup();
    renderWithProviders(<NewConversationButton />);

    await user.click(screen.getByTestId("new-conversation-button"));

    expect(screen.getByTestId("cloud-repo-empty")).toBeInTheDocument();
    expect(screen.queryAllByTestId("launch-repository")).toHaveLength(0);
  });
});
