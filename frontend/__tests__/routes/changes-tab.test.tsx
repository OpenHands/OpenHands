import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router";
import GitChanges from "#/routes/changes-tab";
import { useUnifiedGetGitChanges } from "#/hooks/query/use-unified-get-git-changes";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { ProviderOptions } from "#/types/settings";
import { I18nKey } from "#/i18n/declaration";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      if (key === I18nKey.TIPS$SAVE_WORK) {
        return "Be sure to regularly save your work, either by pushing to GitHub or by downloading your files via VS Code.";
      }

      return key;
    },
  }),
}));

vi.mock("#/hooks/query/use-unified-get-git-changes");
vi.mock("#/hooks/use-agent-state");
vi.mock("#/hooks/query/use-active-conversation");
vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "test-id" }),
}));

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <MemoryRouter>
    <QueryClientProvider client={new QueryClient()}>
      {children}
    </QueryClientProvider>
  </MemoryRouter>
);

describe("Changes Tab", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("should show EmptyChangesMessage when there are no changes", () => {
    vi.mocked(useUnifiedGetGitChanges).mockReturnValue({
      data: [],
      isLoading: false,
      isFetching: false,
      isSuccess: true,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.RUNNING,
    });
    vi.mocked(useActiveConversation).mockReturnValue({
      data: {
        git_provider: null,
      },
    } as ReturnType<typeof useActiveConversation>);

    render(<GitChanges />, { wrapper });

    expect(screen.getByText("DIFF_VIEWER$NO_CHANGES")).toBeInTheDocument();
  });

  it("should not show EmptyChangesMessage when there are changes", () => {
    vi.mocked(useUnifiedGetGitChanges).mockReturnValue({
      data: [{ path: "src/file.ts", status: "M" }],
      isLoading: false,
      isFetching: false,
      isSuccess: true,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.RUNNING,
    });
    vi.mocked(useActiveConversation).mockReturnValue({
      data: {
        git_provider: ProviderOptions.github,
      },
    } as ReturnType<typeof useActiveConversation>);

    render(<GitChanges />, { wrapper });

    expect(
      screen.queryByText("DIFF_VIEWER$NO_CHANGES"),
    ).not.toBeInTheDocument();
  });

  it("shows the GitHub tip when the active conversation uses GitHub", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.75);

    vi.mocked(useUnifiedGetGitChanges).mockReturnValue({
      data: [],
      isLoading: false,
      isFetching: false,
      isSuccess: true,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.RUNNING,
    });
    vi.mocked(useActiveConversation).mockReturnValue({
      data: {
        git_provider: ProviderOptions.github,
      },
    } as ReturnType<typeof useActiveConversation>);

    render(<GitChanges />, { wrapper });

    expect(screen.getByText(I18nKey.TIPS$GITHUB_HOOK)).toBeInTheDocument();
  });

  it("does not show the GitHub tip when the active conversation uses GitLab", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.75);

    vi.mocked(useUnifiedGetGitChanges).mockReturnValue({
      data: [],
      isLoading: false,
      isFetching: false,
      isSuccess: true,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.RUNNING,
    });
    vi.mocked(useActiveConversation).mockReturnValue({
      data: {
        git_provider: ProviderOptions.gitlab,
      },
    } as ReturnType<typeof useActiveConversation>);

    render(<GitChanges />, { wrapper });

    expect(
      screen.queryByText(I18nKey.TIPS$GITHUB_HOOK),
    ).not.toBeInTheDocument();
  });

  it("replaces GitHub with the active provider name in the save-work tip", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.35);

    vi.mocked(useUnifiedGetGitChanges).mockReturnValue({
      data: [],
      isLoading: false,
      isFetching: false,
      isSuccess: true,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.RUNNING,
    });
    vi.mocked(useActiveConversation).mockReturnValue({
      data: {
        git_provider: ProviderOptions.gitlab,
      },
    } as ReturnType<typeof useActiveConversation>);

    render(<GitChanges />, { wrapper });

    expect(
      screen.getByText(/pushing to GitLab/i),
    ).toBeInTheDocument();
    expect(screen.queryByText(/pushing to GitHub/i)).not.toBeInTheDocument();
  });
});
