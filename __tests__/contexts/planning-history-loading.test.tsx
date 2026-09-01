import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ConversationWebSocketProvider } from "#/contexts/conversation-websocket-context";
import { useConversationWebSocket } from "#/contexts/conversation-websocket-context";
import { useEventStore } from "#/stores/use-event-store";
import useMetricsStore from "#/stores/metrics-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { useBrowserStore } from "#/stores/browser-store";
import { useCommandStore } from "#/stores/command-store";
import { useErrorMessageStore } from "#/stores/error-message-store";
import EventService from "#/api/event-service/event-service.api";
import { createUserMessageEvent } from "test-utils";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

vi.mock("#/hooks/use-websocket", () => ({
  useWebSocket: vi.fn(() => ({ socket: null, reconnect: vi.fn() })),
}));
vi.mock("#/hooks/query/use-user-conversation", () => ({
  useUserConversation: vi.fn(() => ({
    data: { conversation_url: "http://localhost/api", session_api_key: null },
  })),
}));

function makePlanningConversation(
  overrides: Partial<AppConversation> = {},
): AppConversation {
  return {
    id: "planning-1",
    created_by_user_id: null,
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    title: "Planner",
    trigger: null,
    pr_number: [],
    llm_model: null,
    metrics: null,
    created_at: "2026-07-28T00:00:00Z",
    updated_at: "2026-07-28T00:00:00Z",
    execution_status: null,
    conversation_url: "http://planner.example/api/conversations/planning-1",
    session_api_key: null,
    sandbox_id: null,
    sub_conversation_ids: [],
    ...overrides,
  };
}

function HistoryProbe() {
  const ctx = useConversationWebSocket();
  return (
    <div data-testid="history-loading">{String(ctx?.isLoadingHistory)}</div>
  );
}

describe("Planning history loading — regression for infinite skeleton (issue #16876)", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    useEventStore.setState({
      events: [],
      eventIds: new Set(),
      uiEvents: [],
      loadedConversationId: null,
    });
    useOptimisticUserMessageStore.setState({ pendingMessages: [] });
    useBrowserStore.getState().reset();
    useMetricsStore.getState().resetMetrics();
    useCommandStore.setState({ commands: [] });
    useErrorMessageStore.getState().removeErrorMessage();
    window.localStorage.clear();
    vi.spyOn(EventService, "searchEvents").mockImplementation(
      async (conversationId: string) => ({
        items: [createUserMessageEvent(`user-msg-` + conversationId)],
        next_page_id: null,
      }),
    );
  });

  it("stays loading while sub-conversations fetch is pending", async () => {
    const { getByTestId } = render(
      <QueryClientProvider client={queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-pending"
          conversationUrl="http://localhost/api"
          subConversationIds={["planning-1"]}
          subConversations={[]}
          isSubConversationsLoading={true}
        >
          <HistoryProbe />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );
    await waitFor(() =>
      expect(getByTestId("history-loading").textContent).toBe("true"),
    );
  });

  it("settles to not loading when fetch settles with no usable planning conversation (empty)", async () => {
    const { getByTestId } = render(
      <QueryClientProvider client={queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-empty"
          conversationUrl="http://localhost/api"
          subConversationIds={["planning-1"]}
          subConversations={[]}
          isSubConversationsLoading={false}
        >
          <HistoryProbe />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );
    await waitFor(() =>
      expect(getByTestId("history-loading").textContent).toBe("false"),
    );
  });

  it("settles when batch fetch error leaves subConversations undefined", async () => {
    const { getByTestId } = render(
      <QueryClientProvider client={queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-error"
          conversationUrl="http://localhost/api"
          subConversationIds={["planning-1"]}
          subConversations={undefined}
          isSubConversationsLoading={false}
        >
          <HistoryProbe />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );
    await waitFor(() =>
      expect(getByTestId("history-loading").textContent).toBe("false"),
    );
  });

  it("settles when sub-conversation entry has no conversation_url", async () => {
    const bad = makePlanningConversation({
      conversation_url: null as unknown as string,
    });
    const { getByTestId } = render(
      <QueryClientProvider client={queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-bad-url"
          conversationUrl="http://localhost/api"
          subConversationIds={["planning-1"]}
          subConversations={[bad]}
          isSubConversationsLoading={false}
        >
          <HistoryProbe />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );
    await waitFor(() =>
      expect(getByTestId("history-loading").textContent).toBe("false"),
    );
  });

  it("remains loading when a usable planning conversation exists and WS has not yet settled", async () => {
    const planning = makePlanningConversation();
    const { getByTestId } = render(
      <QueryClientProvider client={queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-usable"
          conversationUrl="http://localhost/api"
          subConversationIds={[planning.id]}
          subConversations={[planning]}
          isSubConversationsLoading={false}
        >
          <HistoryProbe />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );
    // Usable conversation yields a planning WS URL, so the history gate stays
    // loading until the WS open handler resolves the expected count.
    await waitFor(() =>
      expect(getByTestId("history-loading").textContent).toBe("true"),
    );
  });
});
