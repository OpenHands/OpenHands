import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { ConversationOverviewPanel } from "#/components/features/conversation/conversation-overview-panel";
import { NavigationProvider } from "#/context/navigation-context";
import { ConversationOverviewDrawerProvider } from "#/components/features/conversation/conversation-overview-drawer-context";
import { CONVERSATION_OVERVIEW_DRAWER_SECTION } from "#/components/features/conversation/conversation-overview-drawer.types";

const openSection = vi.fn();
const closeDrawer = vi.fn();
const navigateToCommits = vi.fn();

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "conv-1" }),
}));

vi.mock("#/hooks/use-conversation-overview-git-diff-stats", () => ({
  useConversationOverviewGitDiffStats: () => ({
    additions: 4161,
    deletions: 1824,
    changeCount: 3,
    isLoading: false,
    isError: false,
  }),
}));

vi.mock("#/hooks/use-select-conversation-tab", () => ({
  useSelectConversationTab: () => ({
    navigateToTab: vi.fn(),
    navigateToChanges: vi.fn(),
    navigateToCommits,
  }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      id: "conv-1",
      selected_workspace: "/workspace/project/demo",
      llm_model: "openhands/test-model",
    },
  }),
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({
    data: {
      llm_model: "openhands/test-model",
      agent_settings: {
        mcp_config: {
          mcpServers: {
            example: {
              url: "https://example.com/mcp",
            },
          },
        },
      },
    },
  }),
}));

vi.mock("#/hooks/use-conversation-primary-repository", () => ({
  useConversationPrimaryRepository: () => ({
    repository: "openhands/agent-canvas",
    provider: "github" as const,
    branch: "main",
    isConnected: true,
  }),
}));

vi.mock("#/hooks/query/use-unified-git-commits", () => ({
  useUnifiedGitCommits: () => ({
    commits: [{ sha: "abc" }, { sha: "def" }, { sha: "ghi" }],
    hasMore: false,
    isUnsupported: false,
    isLoading: false,
    isFetching: false,
    isSuccess: true,
    isError: false,
  }),
}));

vi.mock("#/hooks/query/use-repository-git-items", () => ({
  useRepositoryPullRequests: () => ({
    data: [
      {
        id: 1,
        number: 10,
        title: "Fix overview",
        url: "https://github.com/openhands/agent-canvas/pull/10",
        authorLogin: "dev",
        updatedAt: null,
      },
    ],
    isLoading: false,
    isError: false,
  }),
  useRepositoryIssues: () => ({
    data: [],
    isLoading: false,
    isError: false,
  }),
}));

vi.mock("#/api/conversation-metadata-store", () => ({
  getStoredConversationMetadata: () => ({
    selected_workspace: "/workspace/project/demo",
  }),
}));

vi.mock(
  "#/components/features/conversation/conversation-overview-drawer-context",
  async (importOriginal) => {
    const actual = await importOriginal<
      typeof import("#/components/features/conversation/conversation-overview-drawer-context")
    >();
    return {
      ...actual,
      useConversationOverviewDrawerOptional: () => ({
        section: null,
        openAdd: false,
        openSection,
        closeDrawer,
      }),
    };
  },
);

function renderPanel() {
  return render(
    <NavigationProvider
      value={{
        currentPath: "/conversations/conv-1",
        conversationId: "conv-1",
        isNavigating: false,
        navigate: vi.fn(),
      }}
    >
      <ConversationOverviewDrawerProvider>
        <ConversationOverviewPanel />
      </ConversationOverviewDrawerProvider>
    </NavigationProvider>,
  );
}

describe("ConversationOverviewPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  it("renders workspace and git changes without MCP, secrets, skills, or automations", () => {
    renderPanel();

    expect(screen.getByTestId("conversation-overview-panel")).toBeInTheDocument();
    expect(screen.getByTestId("conversation-overview-workspace")).toBeInTheDocument();
    expect(
      screen.queryByTestId("conversation-overview-git-title"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("conversation-overview-diffs")).toBeInTheDocument();
    expect(
      screen.queryByTestId("conversation-overview-mcp"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("conversation-overview-automations"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("conversation-overview-skills"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("conversation-overview-secrets"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("conversation-overview-issues"),
    ).not.toBeInTheDocument();
  });

  it("shows changes inside the git area with commits and pull requests when a repo is connected", async () => {
    const user = userEvent.setup();
    renderPanel();

    const gitBlock = screen.getByTestId("conversation-overview-git-block");
    const diffs = screen.getByTestId("conversation-overview-diffs");
    expect(gitBlock).toContainElement(diffs);

    const workspace = screen.getByTestId("conversation-overview-workspace");
    const gitSection = screen.getByTestId("conversation-overview-git-section");
    // Workspace sits below the git content.
    expect(gitSection.compareDocumentPosition(workspace)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
    const repoLink = screen.getByTestId("conversation-overview-git-repo");
    expect(repoLink).toHaveTextContent("openhands/agent-canvas");
    expect(repoLink.getAttribute("href")).toContain("github.com");
    const branchLink = screen.getByTestId("conversation-overview-git-branch");
    expect(branchLink).toHaveTextContent("main");
    expect(branchLink).toHaveAttribute(
      "href",
      "https://github.com/openhands/agent-canvas/tree/main",
    );
    expect(
      screen.getByTestId("conversation-overview-commits-count"),
    ).toHaveTextContent("3");
    expect(
      screen.getByTestId("conversation-overview-pull-requests-count"),
    ).toHaveTextContent("1");

    await user.click(screen.getByTestId("conversation-overview-commits"));
    expect(navigateToCommits).toHaveBeenCalled();

    await user.click(screen.getByTestId("conversation-overview-pull-requests"));
    expect(openSection).toHaveBeenCalledWith(
      CONVERSATION_OVERVIEW_DRAWER_SECTION.pull_requests,
    );
  });

  it("lets users pin and unpin git changes from the overflow menu", async () => {
    const user = userEvent.setup();
    renderPanel();

    expect(screen.getByTestId("conversation-overview-diffs")).toBeInTheDocument();

    await user.click(screen.getByTestId("conversation-overview-ellipsis"));
    expect(
      screen.getByTestId("conversation-overview-context-menu"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("conversation-overview-menu-divider-git"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("conversation-overview-menu-pin-git-changes"),
    ).toHaveAttribute("aria-pressed", "true");

    await user.click(
      screen.getByTestId("conversation-overview-menu-pin-git-changes"),
    );

    expect(
      screen.queryByTestId("conversation-overview-diffs"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("conversation-overview-menu-pin-git-changes"),
    ).toHaveAttribute("aria-pressed", "false");

    await user.click(
      screen.getByTestId("conversation-overview-menu-pin-git-changes"),
    );

    expect(screen.getByTestId("conversation-overview-diffs")).toBeInTheDocument();
    expect(
      screen.getByTestId("conversation-overview-menu-pin-git-changes"),
    ).toHaveAttribute("aria-pressed", "true");
  });

  it("lets users unpin the git section and individual git parts from the overflow menu", async () => {
    const user = userEvent.setup();
    renderPanel();

    expect(
      screen.getByTestId("conversation-overview-git-section"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("conversation-overview-ellipsis"));
    expect(
      screen.getByTestId("conversation-overview-menu-pin-git"),
    ).toHaveAttribute("aria-pressed", "true");
    expect(
      screen.getByTestId("conversation-overview-menu-pin-git-branch"),
    ).toHaveAttribute("aria-pressed", "true");

    await user.click(
      screen.getByTestId("conversation-overview-menu-pin-git-branch"),
    );
    expect(
      screen.queryByTestId("conversation-overview-git-branch"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("conversation-overview-git-repo"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("conversation-overview-menu-pin-git"));
    expect(
      screen.queryByTestId("conversation-overview-git-block"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("conversation-overview-menu-pin-git"),
    ).toHaveAttribute("aria-pressed", "false");
  });
});
