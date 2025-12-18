import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { AgentStatus } from "#/components/features/controls/agent-status";
import { AgentState } from "#/types/agent-state";
import { useAgentState } from "#/hooks/use-agent-state";
import { useConversationStore } from "#/state/conversation-store";

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

const renderAgentStatus = ({
  isPausing = false,
}: { isPausing?: boolean } = {}) =>
  render(
    <AgentStatus
      handleStop={vi.fn()}
      handleResumeAgent={vi.fn()}
      isPausing={isPausing}
    />,
    { wrapper },
  );

describe("AgentStatus - isLoading logic", () => {
  it("should show loading when curAgentState is INIT", () => {
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.INIT,
    });

    renderAgentStatus();

    expect(screen.getByTestId("agent-loading-spinner")).toBeInTheDocument();
  });

  it("should show loading when isPausing is true, even if shouldShownAgentLoading is false", () => {
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.AWAITING_USER_INPUT,
    });

    renderAgentStatus({ isPausing: true });

    expect(screen.getByTestId("agent-loading-spinner")).toBeInTheDocument();
  });

  it("should NOT update global shouldShownAgentLoading when only isPausing is true", () => {
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.AWAITING_USER_INPUT,
    });

    renderAgentStatus({ isPausing: true });

    // Loading spinner shows (because isPausing)
    expect(screen.getByTestId("agent-loading-spinner")).toBeInTheDocument();

    // But global state should be false (because shouldShownAgentLoading is false)
    const { shouldShownAgentLoading } = useConversationStore.getState();
    expect(shouldShownAgentLoading).toBe(false);
  });
});
