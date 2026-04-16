import { render, screen } from "@testing-library/react";
import { beforeEach, describe, it, expect, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { AgentStatus } from "#/components/features/controls/agent-status";
import { AgentState } from "#/types/agent-state";
import { useAgentState } from "#/hooks/use-agent-state";
import { useConversationStore } from "#/stores/conversation-store";
import { useConversationWebSocket } from "#/contexts/conversation-websocket-context";

vi.mock("#/hooks/use-agent-state");

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "test-id" }),
}));

vi.mock("#/hooks/use-unified-websocket-status", () => ({
  useUnifiedWebSocketStatus: () => "CONNECTED",
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      conversation_id: "test-id",
      status: "RUNNING",
      runtime_status: "STATUS$READY",
      conversation_version: "V1",
    },
  }),
}));

vi.mock("#/hooks/query/use-task-polling", () => ({
  useTaskPolling: () => ({
    taskStatus: null,
    taskDetail: null,
    isTask: false,
  }),
}));

vi.mock("#/hooks/query/use-sub-conversation-task-polling", () => ({
  useSubConversationTaskPolling: () => ({ taskStatus: null }),
}));

vi.mock("#/contexts/conversation-websocket-context", () => ({
  useConversationWebSocket: vi.fn(),
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
  beforeEach(() => {
    vi.mocked(useConversationWebSocket).mockReturnValue(null);
  });

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

  it("should show LOADING_CONVERSATION when V1 history is loading and agent is INIT", () => {
    vi.mocked(useConversationWebSocket).mockReturnValue({
      connectionState: "OPEN",
      sendMessage: vi.fn(),
      isLoadingHistory: true,
    });
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.INIT,
    });

    renderAgentStatus();

    expect(
      screen.getByText("CHAT_INTERFACE$LOADING_CONVERSATION"),
    ).toBeInTheDocument();
  });

  it("should show INITIALIZING when history is not loading and agent is INIT", () => {
    vi.mocked(useConversationWebSocket).mockReturnValue({
      connectionState: "OPEN",
      sendMessage: vi.fn(),
      isLoadingHistory: false,
    });
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.INIT,
    });

    renderAgentStatus();

    expect(screen.getByText("AGENT_STATUS$INITIALIZING")).toBeInTheDocument();
  });
});
