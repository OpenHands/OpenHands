import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, waitFor, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createUserMessageEvent } from "test-utils";
import { ConversationWebSocketProvider } from "#/contexts/conversation-websocket-context";
import { useEventStore } from "#/stores/use-event-store";
import useMetricsStore from "#/stores/metrics-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { useBrowserStore } from "#/stores/browser-store";
import { useCommandStore } from "#/stores/command-store";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { useUserConversation } from "#/hooks/query/use-user-conversation";
import { useWebSocket } from "#/hooks/use-websocket";
import EventService from "#/api/event-service/event-service.api";
import {
  getStoredConversationMetadata,
  setStoredConversationMetadata,
} from "#/api/conversation-metadata-store";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { MessageEvent } from "#/types/agent-server/core";
import { isStreamingDeltaEvent } from "#/types/agent-server/type-guards";

type CapturedWebSocketOptions = {
  onMessage?: (event: { data: string }) => void;
  queryParams?: Record<string, string | boolean>;
  sessionApiKey?: string | null;
};

const wsCapture = vi.hoisted(() => ({
  mainOnMessage: null as null | ((event: { data: string }) => void),
  mainOptions: null as CapturedWebSocketOptions | null,
  planningOnMessage: null as null | ((event: { data: string }) => void),
  calls: [] as Array<{
    url: string;
    options?: CapturedWebSocketOptions;
  }>,
}));

const errorHandlerMocks = vi.hoisted(() => ({
  trackError: vi.fn(),
}));

// Keep the units under test real (the provider, `useConversationHistory`, the
// event store). Only the network is stubbed: the WebSocket transport and the
// REST service the history query depends on.
vi.mock("#/hooks/use-websocket", () => ({
  useWebSocket: vi.fn((url: string, options?: CapturedWebSocketOptions) => {
    if (url) {
      wsCapture.calls.push({ url, options });
    }
    if (
      url &&
      options?.onMessage &&
      options.queryParams &&
      "resend_mode" in options.queryParams
    ) {
      wsCapture.mainOnMessage = options.onMessage;
      wsCapture.mainOptions = options;
    }
    if (
      url &&
      options?.onMessage &&
      options.queryParams &&
      "resend_all" in options.queryParams
    ) {
      wsCapture.planningOnMessage = options.onMessage;
    }
    return { socket: null, reconnect: vi.fn() };
  }),
}));
vi.mock("#/hooks/query/use-user-conversation", () => ({
  useUserConversation: vi.fn(),
}));
vi.mock("#/utils/error-handler", () => ({
  trackError: errorHandlerMocks.trackError,
}));

const AGENT_REPLY_ID = "evt-agent-reply";

// An agent reply that streamed in over the WebSocket *after* the initial REST
// history page — i.e. it lives only in the event store, never in the cached
// history page. This is the class of event the old code dropped on re-entry.
const makeAgentReply = (): MessageEvent => ({
  id: AGENT_REPLY_ID,
  timestamp: new Date(Date.now() + 1000).toISOString(),
  source: "agent",
  llm_message: { role: "assistant", content: [{ type: "text", text: "Hi!" }] },
  activated_skills: [],
  extended_content: [],
});

const eventIds = () => useEventStore.getState().events.map((event) => event.id);

describe("ConversationWebSocketProvider — conversation-scoped event store", () => {
  let queryClient: QueryClient;

  const renderProvider = (conversationId: string) =>
    render(
      <QueryClientProvider client={queryClient}>
        <ConversationWebSocketProvider
          conversationId={conversationId}
          conversationUrl={null}
        >
          <div />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );

  beforeEach(() => {
    wsCapture.mainOnMessage = null;
    wsCapture.mainOptions = null;
    wsCapture.planningOnMessage = null;
    wsCapture.calls.length = 0;
    window.localStorage.clear();
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

    vi.mocked(useUserConversation).mockReturnValue({
      data: { conversation_url: "http://localhost/api", session_api_key: null },
    } as ReturnType<typeof useUserConversation>);

    // The cached REST history page ends at the user's message — a fresh page
    // per conversation so we can detect cross-conversation leakage.
    vi.spyOn(EventService, "searchEvents").mockImplementation(
      async (conversationId: string) => ({
        items: [createUserMessageEvent(`user-msg-${conversationId}`)],
        next_page_id: null,
      }),
    );
  });

  afterEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
  });

  // A successful model switch the agent performed on its own (via the
  // SwitchLLM tool), delivered over the main WebSocket.
  const makeAgentSwitchObservation = (profileName: string) => ({
    id: "evt-switch-1",
    timestamp: new Date().toISOString(),
    source: "environment",
    action_id: "action-switch-1",
    tool_name: "switch_llm",
    tool_call_id: "call-switch-1",
    observation: {
      kind: "SwitchLLMObservation",
      content: [{ type: "text", text: `Switched to ${profileName}` }],
      is_error: false,
      profile_name: profileName,
      reason: null,
      active_model: null,
    },
  });

  it("stamps active_profile on a successful agent-triggered model switch so it survives reload", async () => {
    // Arrange: open a conversation with a real ws url so the main socket's
    // onMessage (handleMainMessage) is wired and captured.
    render(
      <QueryClientProvider client={queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-switch"
          conversationUrl="http://localhost/api"
        >
          <div />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );
    await waitFor(() => expect(wsCapture.mainOnMessage).not.toBeNull());

    // Act: the agent switches to "fast-opus" via the SwitchLLM tool.
    act(() => {
      wsCapture.mainOnMessage!({
        data: JSON.stringify(makeAgentSwitchObservation("fast-opus")),
      });
    });

    // Assert: the profile identity is persisted to stored metadata — the same
    // field the chat-header switcher reads after a reload (#1082). Without the
    // stamp this stays null and the header falls back to ambiguous matching.
    expect(getStoredConversationMetadata("conv-switch")?.active_profile).toBe(
      "fast-opus",
    );
  });

  it("keeps the session key out of WebSocket query parameters", async () => {
    const sessionApiKey = `sk-oh-${"c".repeat(64)}`;

    render(
      <QueryClientProvider client={queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-auth"
          conversationUrl="http://localhost/api"
          sessionApiKey={sessionApiKey}
        >
          <div />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );

    await waitFor(() => expect(wsCapture.mainOptions).not.toBeNull());

    expect(wsCapture.mainOptions?.sessionApiKey).toBe(sessionApiKey);
    expect(wsCapture.mainOptions?.queryParams).not.toHaveProperty(
      "session_api_key",
    );
  });

  it("keeps the events socket up, with its `since` anchor, across background history refetches", async () => {
    // Arrange: the initial history load resolves; the background refetch stays
    // in flight so the query sits in `isFetching` while the socket is already
    // established — the state that used to tear the socket down and leave the
    // conversation stuck at "Connecting".
    const historyPage = () => ({
      items: [createUserMessageEvent("user-msg-conv-refetch")],
      next_page_id: null,
    });
    let resolveRefetch!: (
      page: Awaited<ReturnType<typeof EventService.searchEvents>>,
    ) => void;
    vi.spyOn(EventService, "searchEvents")
      .mockResolvedValueOnce(historyPage())
      .mockImplementationOnce(
        () =>
          new Promise<Awaited<ReturnType<typeof EventService.searchEvents>>>(
            (resolve) => {
              resolveRefetch = resolve;
            },
          ),
      );

    render(
      <QueryClientProvider client={queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-refetch"
          conversationUrl="http://localhost/api"
        >
          <div />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );
    await waitFor(() => expect(wsCapture.mainOptions).not.toBeNull());

    // Every render's main-socket call (the one carrying `resend_mode`),
    // including any teardown call with an empty URL.
    const mainCalls = () =>
      vi
        .mocked(useWebSocket)
        .mock.calls.filter(
          ([, options]) =>
            options?.queryParams && "resend_mode" in options.queryParams,
        );
    const connectedAt = mainCalls().length;
    const anchor = wsCapture.mainOptions?.queryParams?.after_timestamp;
    expect(anchor).toBeTruthy();

    // Act: a background refetch starts (as `refetchOnMount: "always"` fires
    // when returning to a conversation) and stays in flight.
    act(() => {
      void queryClient.refetchQueries({ queryKey: ["conversation-history"] });
    });
    await waitFor(() =>
      expect(
        queryClient.isFetching({ queryKey: ["conversation-history"] }),
      ).toBe(1),
    );

    // Assert: since the socket connected, no render tore it down (empty URL)
    // and none degraded the `since` anchor to a full resend.
    for (const [url, options] of mainCalls().slice(connectedAt - 1)) {
      expect(url).toContain("/sockets/events/conv-refetch");
      expect(options?.queryParams).toMatchObject({
        resend_mode: "since",
        after_timestamp: anchor,
      });
    }

    // The refetch settling must not churn the socket either.
    await act(async () => {
      resolveRefetch(historyPage());
    });
    const [urlAfterRefetch] = mainCalls().at(-1)!;
    expect(urlAfterRefetch).toContain("/sockets/events/conv-refetch");
  });

  it("uses the planning sub-conversation session key", async () => {
    const mainSessionApiKey = `sk-oh-main-${"m".repeat(48)}`;
    const planningSessionApiKey = `sk-oh-plan-${"p".repeat(48)}`;
    const planningConversation: AppConversation = {
      id: "planning-auth",
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
      conversation_url:
        "http://planner.example/api/conversations/planning-auth",
      session_api_key: planningSessionApiKey,
      sandbox_id: null,
      sub_conversation_ids: [],
    };

    render(
      <QueryClientProvider client={queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-auth"
          conversationUrl="http://main.example/api/conversations/conv-auth"
          sessionApiKey={mainSessionApiKey}
          subConversationIds={[planningConversation.id]}
          subConversations={[planningConversation]}
        >
          <div />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );

    await waitFor(() =>
      expect(
        wsCapture.calls.some(({ url }) =>
          url.endsWith("/sockets/events/planning-auth"),
        ),
      ).toBe(true),
    );

    const planningCall = wsCapture.calls.find(({ url }) =>
      url.endsWith("/sockets/events/planning-auth"),
    );

    expect(planningCall?.url).toBe(
      "ws://planner.example/sockets/events/planning-auth",
    );
    expect(planningCall?.options?.sessionApiKey).toBe(planningSessionApiKey);
    expect(planningCall?.options?.queryParams).toEqual({ resend_all: true });
    expect(planningCall?.options?.queryParams).not.toHaveProperty(
      "session_api_key",
    );
  });

  it("preserves the conversation's attached plugins across an agent-triggered model switch", async () => {
    // Arrange: the conversation's metadata already carries an attached plugin.
    setStoredConversationMetadata("conv-switch", {
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
      plugins: [
        { source: "github:acme/city-weather", ref: null, repo_path: null },
      ],
    });
    render(
      <QueryClientProvider client={queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-switch"
          conversationUrl="http://localhost/api"
        >
          <div />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );
    await waitFor(() => expect(wsCapture.mainOnMessage).not.toBeNull());

    // Act: the agent switches model via the SwitchLLM tool.
    act(() => {
      wsCapture.mainOnMessage!({
        data: JSON.stringify(makeAgentSwitchObservation("fast-opus")),
      });
    });

    // Assert: the plugins snapshot survives the full-object metadata replace.
    expect(getStoredConversationMetadata("conv-switch")?.plugins).toEqual([
      { source: "github:acme/city-weather", ref: null, repo_path: null },
    ]);
  });

  // On reconnect the backlog is replayed; non-idempotent side-effects must not
  // fire again for events already processed (#1656).
  describe("reconnect replay does not re-run non-idempotent side-effects", () => {
    const makeBashAction = (id: string, command: string) => ({
      id,
      timestamp: new Date().toISOString(),
      source: "agent",
      thought: [],
      thinking_blocks: [],
      action: {
        kind: "ExecuteBashAction",
        command,
        is_input: false,
        timeout: null,
        reset: false,
      },
      tool_name: "execute_bash",
      tool_call_id: `call-${id}`,
      tool_call: {
        id: `call-${id}`,
        type: "function",
        function: {
          name: "execute_bash",
          arguments: JSON.stringify({ command }),
        },
      },
      llm_response_id: `resp-${id}`,
      security_risk: "UNKNOWN",
    });

    const makeBashObservation = (
      id: string,
      actionId: string,
      text: string,
    ) => ({
      id,
      timestamp: new Date().toISOString(),
      source: "environment",
      action_id: actionId,
      tool_name: "execute_bash",
      tool_call_id: `call-${actionId}`,
      observation: {
        kind: "ExecuteBashObservation",
        content: [{ type: "text", text }],
        command: "run",
        exit_code: 0,
        error: false,
        timeout: false,
        metadata: {
          exit_code: 0,
          pid: 1,
          username: "u",
          hostname: "h",
          working_dir: "/",
          py_interpreter_path: null,
          prefix: "",
          suffix: "",
        },
      },
    });

    const makeConversationError = (
      id: string,
      detail: string,
      classification?: {
        kind: "auth";
        retryable: boolean;
        user_action: "settings";
      },
    ) => ({
      id,
      timestamp: new Date().toISOString(),
      source: "environment",
      kind: "ConversationErrorEvent",
      detail,
      code: "SomeError",
      ...(classification ? { classification } : {}),
    });

    const makeAgentError = (
      id: string,
      classification?: {
        kind: "auth";
        retryable: boolean;
        user_action: "settings";
      },
    ) => ({
      id,
      timestamp: new Date().toISOString(),
      source: "agent",
      message_id: `msg-${id}`,
      message_seq: 1,
      error: "Agent failed",
      error_type: "AgentError",
      tool_name: "generic",
      tool_call_id: `call-${id}`,
      ...(classification ? { classification } : {}),
    });

    const renderCaptured = async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <ConversationWebSocketProvider
            conversationId="conv-reconnect"
            conversationUrl="http://localhost/api"
          >
            <div />
          </ConversationWebSocketProvider>
        </QueryClientProvider>,
      );
      await waitFor(() => expect(wsCapture.mainOnMessage).not.toBeNull());
    };

    const deliver = (event: unknown) =>
      act(() => {
        wsCapture.mainOnMessage!({ data: JSON.stringify(event) });
      });

    it("does not re-append terminal input/output for replayed bash events", async () => {
      await renderCaptured();

      const action = makeBashAction("bash-action-1", "echo hi");
      const observation = makeBashObservation(
        "bash-obs-1",
        "bash-action-1",
        "hi\n",
      );

      // First delivery, then a reconnect replay of the same two events.
      deliver(action);
      deliver(observation);
      deliver(action);
      deliver(observation);

      expect(useCommandStore.getState().commands).toEqual([
        { content: "echo hi", type: "input" },
        { content: "hi\n", type: "output" },
      ]);
    });

    it("does not re-raise a dismissed error banner when the error event is replayed", async () => {
      await renderCaptured();

      const errorEvent = makeConversationError("conv-error-1", "Boom");

      // Show the banner, dismiss it, then replay the error on reconnect.
      deliver(errorEvent);
      expect(useErrorMessageStore.getState().errorMessage).toBe("Boom");
      act(() => useErrorMessageStore.getState().removeErrorMessage());
      expect(useErrorMessageStore.getState().errorMessage).toBeNull();

      // It must stay dismissed.
      deliver(errorEvent);
      expect(useErrorMessageStore.getState().errorMessage).toBeNull();
    });

    it("forwards error classifications to the banner store and telemetry", async () => {
      await renderCaptured();
      const classification = {
        kind: "auth" as const,
        retryable: false,
        user_action: "settings" as const,
      };

      deliver(
        makeConversationError(
          "conv-error-2",
          "Authentication failed",
          classification,
        ),
      );

      expect(useErrorMessageStore.getState().errorClassification).toEqual(
        classification,
      );
      expect(errorHandlerMocks.trackError).toHaveBeenCalledWith({
        source: "conversation",
        metadata: {
          eventId: "conv-error-2",
          errorCode: "SomeError",
        },
        classification,
      });
    });

    it("forwards AgentErrorEvent classifications to telemetry (main agent)", async () => {
      await renderCaptured();
      const classification = {
        kind: "auth" as const,
        retryable: false,
        user_action: "settings" as const,
      };

      deliver(makeAgentError("agent-err-1", classification));

      expect(errorHandlerMocks.trackError).toHaveBeenCalledWith({
        source: "agent",
        metadata: {
          eventId: "agent-err-1",
          toolName: "generic",
          toolCallId: "call-agent-err-1",
        },
        classification,
      });
    });

    it("forwards AgentErrorEvent classifications to telemetry (planning agent)", async () => {
      const planningConversation: AppConversation = {
        id: "planning-err",
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
        conversation_url:
          "http://planner.example/api/conversations/planning-err",
        session_api_key: null,
        sandbox_id: null,
        sub_conversation_ids: [],
      };
      const classification = {
        kind: "auth" as const,
        retryable: true,
        user_action: "settings" as const,
      };

      render(
        <QueryClientProvider client={queryClient}>
          <ConversationWebSocketProvider
            conversationId="conv-err"
            conversationUrl="http://main.example/api/conversations/conv-err"
            subConversationIds={[planningConversation.id]}
            subConversations={[planningConversation]}
          >
            <div />
          </ConversationWebSocketProvider>
        </QueryClientProvider>,
      );
      // Wait for the planning sub-conversation WebSocket to be established.
      await waitFor(() => expect(wsCapture.planningOnMessage).not.toBeNull());

      act(() => {
        wsCapture.planningOnMessage!({
          data: JSON.stringify(makeAgentError("agent-err-2", classification)),
        });
      });

      expect(errorHandlerMocks.trackError).toHaveBeenCalledWith({
        source: "planning_agent",
        metadata: {
          eventId: "agent-err-2",
          toolName: "generic",
          toolCallId: "call-agent-err-2",
        },
        classification,
      });
    });
  });

  it("clears the previous conversation's events when switching conversations", async () => {
    // Arrange + Act: open conversation A.
    const { rerender } = renderProvider("conv-a");
    await waitFor(() => expect(eventIds()).toEqual(["user-msg-conv-a"]));

    // Act: switch to conversation B.
    rerender(
      <QueryClientProvider client={queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-b"
          conversationUrl={null}
        >
          <div />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );

    // Assert: B's history replaced A's — A did not leak into B.
    await waitFor(() => expect(eventIds()).toEqual(["user-msg-conv-b"]));
  });

  it("resets browser-panel state when switching conversations", async () => {
    const { rerender } = renderProvider("conv-a");
    await waitFor(() => expect(eventIds()).toEqual(["user-msg-conv-a"]));

    useBrowserStore.setState({
      url: "https://example.com",
      screenshotSrc: "data:image/png;base64,abc123",
    });

    rerender(
      <QueryClientProvider client={queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-b"
          conversationUrl={null}
        >
          <div />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );

    await waitFor(() =>
      expect(useBrowserStore.getState().screenshotSrc).toBe(""),
    );
    expect(useBrowserStore.getState().url).toBe("");
  });

  it("resets the metrics store when switching conversations", async () => {
    const { rerender } = renderProvider("conv-a");
    await waitFor(() => expect(eventIds()).toEqual(["user-msg-conv-a"]));

    useMetricsStore.setState({
      cost: 1.5,
      max_budget_per_task: 5,
      usage: {
        prompt_tokens: 10,
        completion_tokens: 20,
        cache_read_tokens: 1,
        cache_write_tokens: 2,
        context_window: 128_000,
        per_turn_token: 500,
      },
    });

    rerender(
      <QueryClientProvider client={queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-b"
          conversationUrl={null}
        >
          <div />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );

    await waitFor(() => expect(useMetricsStore.getState().usage).toBeNull());
    expect(useMetricsStore.getState().cost).toBeNull();
    expect(useMetricsStore.getState().max_budget_per_task).toBeNull();
  });

  it("keeps events that arrived after history when re-entering the same conversation", async () => {
    // Arrange: open conversation A, then receive an agent reply over the socket
    // that is not part of the cached REST history page.
    const { unmount } = renderProvider("conv-a");
    await waitFor(() => expect(eventIds()).toEqual(["user-msg-conv-a"]));
    act(() => {
      useEventStore.getState().addEvent(makeAgentReply());
    });

    // Act: leave (e.g. to Settings) and return to the same conversation.
    unmount();
    renderProvider("conv-a");

    // Assert: both the user message and the streamed reply survive re-entry.
    await waitFor(() =>
      expect(eventIds()).toEqual(["user-msg-conv-a", AGENT_REPLY_ID]),
    );
    // ...and the re-seed deduped against the existing user message rather than
    // appending a second copy — exactly two events, no double-insertion.
    expect(eventIds()).toHaveLength(2);
  });

  const makeStreamingDelta = (id: string, content: string) => ({
    id,
    timestamp: new Date().toISOString(),
    source: "agent",
    kind: "StreamingDeltaEvent",
    content,
    reasoning_content: null,
  });

  const makeAgentMessage = (id: string, text: string): MessageEvent => ({
    id,
    timestamp: new Date(Date.now() + 1000).toISOString(),
    source: "agent",
    llm_message: { role: "assistant", content: [{ type: "text", text }] },
    activated_skills: [],
    extended_content: [],
  });

  const renderProviderWithUrl = (conversationId: string) =>
    render(
      <QueryClientProvider client={queryClient}>
        <ConversationWebSocketProvider
          conversationId={conversationId}
          conversationUrl="http://localhost/api"
        >
          <div />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );

  it("buffers streaming deltas, then flushes them (reconciled) when the final message arrives", async () => {
    renderProviderWithUrl("conv-stream");
    await waitFor(() => expect(wsCapture.mainOnMessage).not.toBeNull());
    await waitFor(() => expect(eventIds()).toEqual(["user-msg-conv-stream"]));

    // Deltas arrive: they are buffered by the batcher, NOT committed per token.
    act(() => {
      wsCapture.mainOnMessage!({
        data: JSON.stringify(makeStreamingDelta("d1", "I'll help")),
      });
      wsCapture.mainOnMessage!({
        data: JSON.stringify(makeStreamingDelta("d2", " with that.")),
      });
    });
    expect(eventIds()).toEqual(["user-msg-conv-stream"]);

    // The final agent message is a non-delta event: the handler flushes the
    // buffered deltas first, so the message reconciles the streamed text in
    // place instead of racing ahead of it.
    act(() => {
      wsCapture.mainOnMessage!({
        data: JSON.stringify(
          makeAgentMessage("agent-final", "I'll help with that. Done."),
        ),
      });
    });

    const { uiEvents, eventIds: ids } = useEventStore.getState();
    // One reconciled agent bubble: the canonical final message supersedes the
    // flushed deltas, so the streamed text renders once and is never duplicated.
    expect(uiEvents).toHaveLength(2);
    const bubble = uiEvents[1] as MessageEvent;
    expect(bubble.id).toBe("agent-final");
    expect(bubble.llm_message.content).toEqual([
      { type: "text", text: "I'll help with that. Done." },
    ]);
    expect(uiEvents.some((event) => isStreamingDeltaEvent(event))).toBe(false);
    // eventIds tracks the two durable events, never the deltas.
    expect(ids.size).toBe(2);
  });

  it("discards buffered deltas from the previous conversation on switch", async () => {
    const { rerender } = renderProviderWithUrl("conv-a");
    await waitFor(() => expect(wsCapture.mainOnMessage).not.toBeNull());
    await waitFor(() => expect(eventIds()).toEqual(["user-msg-conv-a"]));

    // Buffer deltas for A, then switch to B before they flush.
    act(() => {
      wsCapture.mainOnMessage!({
        data: JSON.stringify(makeStreamingDelta("a1", "STALE")),
      });
    });
    rerender(
      <QueryClientProvider client={queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-b"
          conversationUrl="http://localhost/api"
        >
          <div />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );
    await waitFor(() => expect(eventIds()).toEqual(["user-msg-conv-b"]));

    // B streams and finalizes. If the switch had NOT reset the batcher, A's
    // "STALE" delta would still be buffered and merge into B's stream here.
    act(() => {
      wsCapture.mainOnMessage!({
        data: JSON.stringify(makeStreamingDelta("b1", "fresh")),
      });
      wsCapture.mainOnMessage!({
        data: JSON.stringify(makeAgentMessage("agent-b", "fresh.")),
      });
    });

    const { uiEvents, events } = useEventStore.getState();
    expect(uiEvents).toHaveLength(2);
    expect((uiEvents[1] as MessageEvent).llm_message.content).toEqual([
      { type: "text", text: "fresh." },
    ]);
    // The committed delta carries B's text only — had A's buffer survived the
    // switch it would have merged in ahead of it as "STALEfresh".
    const committedDeltas = events.filter((event) =>
      isStreamingDeltaEvent(event),
    );
    expect(committedDeltas.map((delta) => delta.content)).toEqual(["fresh"]);
    expect(JSON.stringify(events)).not.toContain("STALE");
  });

  it("consumes the optimistic pending bubble when the echoed user message arrives via REST preload", async () => {
    // Arrange: a cloud start-task conversation left a "Sending…" bubble whose
    // content matches the first message the server has already persisted. With
    // the WebSocket stubbed, the only path that delivers the echo is the REST
    // history preload — the path that previously left this bubble orphaned.
    useOptimisticUserMessageStore.setState({
      pendingMessages: [
        {
          id: "pending-1",
          conversationId: "conv-a",
          text: "User message",
          content: "User message",
          status: "sending",
          imageUrls: [],
          fileUrls: [],
          timestamp: new Date().toISOString(),
        },
      ],
    });

    // Act: open the conversation; preload returns the echoed user message.
    renderProvider("conv-a");

    // Assert: the preloaded echo cleared the bubble, so it isn't shown twice.
    await waitFor(() =>
      expect(useOptimisticUserMessageStore.getState().pendingMessages).toEqual(
        [],
      ),
    );
  });
});
