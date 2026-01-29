import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router";
import GitChanges from "#/routes/changes-tab";
import { useUnifiedGetGitChanges } from "#/hooks/query/use-unified-get-git-changes";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

vi.mock("#/hooks/query/use-unified-get-git-changes");
vi.mock("#/hooks/use-agent-state");
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
  it("should show EmptyChangesMessage when there are no changes", () => {
    vi.mocked(useUnifiedGetGitChanges).mockReturnValue({
      data: [],
      isLoading: false,
      isSuccess: true,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.RUNNING,
    });

    render(<GitChanges />, { wrapper });

    expect(screen.getByText("DIFF_VIEWER$NO_CHANGES")).toBeInTheDocument();
  });

  it("should not show EmptyChangesMessage when there are changes", () => {
    vi.mocked(useUnifiedGetGitChanges).mockReturnValue({
      data: [{ path: "src/file.ts", status: "M" }],
      isLoading: false,
      isSuccess: true,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.RUNNING,
    });

    render(<GitChanges />, { wrapper });

    expect(
      screen.queryByText("DIFF_VIEWER$NO_CHANGES"),
    ).not.toBeInTheDocument();
  });
});
