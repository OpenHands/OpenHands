import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, waitFor, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { createUserMessageEvent } from "test-utils";
import {
  ConversationWebSocketProvider,
  useConversationWebSocket,
} from "#/contexts/conversation-websocket-context";
import { useConversationStore } from "#/stores/conversation-store";
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

type QueryParams = Record<string, string | boolean>;

type CapturedWebSocketOptions = {
  onMessage?: (event: { data: string }) => void;
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  queryParams?: QueryParams | (() => QueryParams);
  sessionApiKey?: string | null;
};

/** Both sockets resolve their params at connect time (the resume cursor). */
const resolveQueryParams = (
  options?: { queryParams?: QueryParams | (() => QueryParams) } | null,
): QueryParams => {
  const params = options?.queryParams;
  return typeof params === "function" ? params() : (params ?? {});
};

/** Wrap a durable event the way `/sockets/session/{id}` does. */
let nextSeq = 0;
const durable = (event: unknown, seq = nextSeq++) =>
  JSON.stringify({ type: "durable", seq, event });

const wsCapture = vi.hoisted(() => ({
  hookCalls: 0,
  mainOnMessage: null as null | ((event: { data: string }) => void),
  mainOptions: null as CapturedWebSocketOptions | null,
  planningOnMessage: null as null | ((event: { data: string }) => void),
  planningOptions: null as CapturedWebSocketOptions | null,
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
    // Both sockets speak the same protocol now, so neither the URL nor the
    // query params tell them apart. Hooks run in declaration order on every
    // render — main socket first, planning second — so call parity does.
    const isMain = wsCapture.hookCalls % 2 === 0;
    wsCapture.hookCalls += 1;
    if (url && options?.onMessage) {
      if (isMain) {
        wsCapture.mainOnMessage = options.onMessage;
        wsCapture.mainOptions = options;
      } else {
        wsCapture.planningOnMessage = options.onMessage;
        wsCapture.planningOptions = options;
      }
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

const sendEventMock = vi.hoisted(() => vi.fn());
vi.mock("@openhands/typescript-client/clients", async () => {
  const actual = await vi.importActual<
    typeof import("@openhands/typescript-client/clients")
  >("@openhands/typescript-client/clients");
  return {
    ...actual,
    ConversationClient: vi.fn(function ConversationClientMock() {
      return { sendEvent: sendEventMock };
    }),
  };
});

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
    nextSeq = 0;
    wsCapture.hookCalls = 0;
    wsCapture.mainOnMessage = null;
    wsCapture.mainOptions = null;
    wsCapture.planningOnMessage = null;
    wsCapture.planningOptions = null;
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
    // Unfiltered count = log length; the history page turns it into the
    // first-connect cursor.
    vi.spyOn(EventService, "getEventCount").mockResolvedValue(1);
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
        data: durable(makeAgentSwitchObservation("fast-opus")),
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
    expect(resolveQueryParams(wsCapture.mainOptions)).not.toHaveProperty(
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

    // Every render's main-socket call (see the mock: main is the even one),
    // including any teardown call with an empty URL.
    const mainCalls = () =>
      vi.mocked(useWebSocket).mock.calls.filter((_, index) => index % 2 === 0);
    const connectedAt = mainCalls().length;
    expect(resolveQueryParams(wsCapture.mainOptions)).toHaveProperty(
      "after_seq",
    );

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
    // and every call still carries the resume cursor.
    for (const [url, options] of mainCalls().slice(connectedAt - 1)) {
      expect(url).toContain("/sockets/session/conv-refetch");
      expect(resolveQueryParams(options)).toHaveProperty("after_seq");
    }

    // The refetch settling must not churn the socket either.
    await act(async () => {
      resolveRefetch(historyPage());
    });
    const [urlAfterRefetch] = mainCalls().at(-1)!;
    expect(urlAfterRefetch).toContain("/sockets/session/conv-refetch");
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
          url.endsWith("/sockets/session/planning-auth"),
        ),
      ).toBe(true),
    );

    const planningCall = wsCapture.calls.find(({ url }) =>
      url.endsWith("/sockets/session/planning-auth"),
    );

    expect(planningCall?.url).toBe(
      "ws://planner.example/sockets/session/planning-auth",
    );
    expect(planningCall?.options?.sessionApiKey).toBe(planningSessionApiKey);
    // No REST preload for the planner, so it replays the whole log.
    expect(resolveQueryParams(planningCall?.options)).toEqual({
      after_seq: "-1",
    });
    expect(resolveQueryParams(planningCall?.options)).not.toHaveProperty(
      "session_api_key",
    );
  });

  // The socket is never OPEN in these tests (the useWebSocket mock returns
  // `socket: null`), so every send falls through to the REST queue — exactly
  // the window this suite is about.
  describe("plan-mode message routing before the planning socket opens", () => {
    function SendMessageProbe({
      onReady,
    }: {
      onReady: (send: ReturnType<typeof useConversationWebSocket>) => void;
    }) {
      const context = useConversationWebSocket();
      React.useEffect(() => onReady(context), [context, onReady]);
      return null;
    }

    const renderPlanMode = (subConversationIds?: string[]) => {
      let context: ReturnType<typeof useConversationWebSocket> | null = null;
      render(
        <QueryClientProvider client={queryClient}>
          <ConversationWebSocketProvider
            conversationId="conv-parent"
            conversationUrl="http://localhost/api"
            subConversationIds={subConversationIds}
            // The react-query lookup that resolves ids into conversations has
            // not landed yet — this is the race the fix is about.
            subConversations={undefined}
          >
            <SendMessageProbe
              onReady={(value) => {
                context = value;
              }}
            />
          </ConversationWebSocketProvider>
        </QueryClientProvider>,
      );
      return () => context!;
    };

    beforeEach(() => {
      sendEventMock.mockReset().mockResolvedValue(undefined);
      useConversationStore.setState({ conversationMode: "plan" });
    });

    afterEach(() => {
      useConversationStore.setState({ conversationMode: "code" });
    });

    it("queues the first prompt to the planner, not the parent code agent", async () => {
      const getContext = renderPlanMode(["planning-1"]);
      await waitFor(() => expect(getContext()).not.toBeNull());

      await act(async () => {
        await getContext().sendMessage({
          role: "user",
          content: [{ type: "text", text: "plan this" }],
        });
      });

      expect(sendEventMock).toHaveBeenCalledWith(
        "planning-1",
        expect.anything(),
        expect.anything(),
      );
    });

    it("errors instead of falling back to the parent when no planner exists yet", async () => {
      const getContext = renderPlanMode(undefined);
      await waitFor(() => expect(getContext()).not.toBeNull());

      await expect(
        act(async () => {
          await getContext().sendMessage({
            role: "user",
            content: [{ type: "text", text: "plan this" }],
          });
        }),
      ).rejects.toThrow("Planning conversation is not ready yet");

      // Falling back to `conv-parent` here would run a planning prompt in the
      // code agent — the boundary plan mode exists to enforce.
      expect(sendEventMock).not.toHaveBeenCalled();
    });
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
        data: durable(makeAgentSwitchObservation("fast-opus")),
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
        wsCapture.mainOnMessage!({ data: durable(event) });
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
          data: durable(makeAgentError("agent-err-2", classification)),
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

  const itemStarted = (itemId: string, anchorSeq: number | null = null) =>
    JSON.stringify({
      type: "item_started",
      item_id: itemId,
      attempt: 1,
      ...(anchorSeq === null ? {} : { anchor_seq: anchorSeq }),
    });

  const delta = (itemId: string, content: string, order: number) =>
    JSON.stringify({
      type: "delta",
      item_id: itemId,
      attempt: 1,
      order,
      kind: "text",
      content,
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

  it("buffers deltas, then retires the slot by id when the durable message arrives", async () => {
    renderProviderWithUrl("conv-stream");
    await waitFor(() => expect(wsCapture.mainOnMessage).not.toBeNull());
    await waitFor(() => expect(eventIds()).toEqual(["user-msg-conv-stream"]));

    // The slot opens before the first token; deltas are buffered by the
    // batcher, NOT committed per token.
    act(() => {
      wsCapture.mainOnMessage!({ data: itemStarted("agent-final") });
      wsCapture.mainOnMessage!({ data: delta("agent-final", "I'll help", 0) });
      wsCapture.mainOnMessage!({
        data: delta("agent-final", " with that.", 1),
      });
    });
    // Slots live only in `uiEvents`; the durable log is untouched.
    expect(eventIds()).toEqual(["user-msg-conv-stream"]);

    // The durable frame flushes the buffer first, then retires the slot on
    // `event.id === item_id` — one equality test, no text comparison.
    act(() => {
      wsCapture.mainOnMessage!({
        data: durable(makeAgentMessage("agent-final", "I'll help with that.")),
      });
    });

    const { uiEvents, eventIds: ids } = useEventStore.getState();
    expect(uiEvents).toHaveLength(2);
    const bubble = uiEvents[1] as MessageEvent;
    expect(bubble.id).toBe("agent-final");
    expect(uiEvents.some((event) => isStreamingDeltaEvent(event))).toBe(false);
    // eventIds tracks the two durable events, never the slot.
    expect(ids.size).toBe(2);
  });

  it("keeps a user message that lands mid-stream below the bubble (#15433)", async () => {
    renderProviderWithUrl("conv-split");
    await waitFor(() => expect(wsCapture.mainOnMessage).not.toBeNull());
    await waitFor(() => expect(eventIds()).toEqual(["user-msg-conv-split"]));

    act(() => {
      wsCapture.mainOnMessage!({ data: itemStarted("agent-split", 0) });
      wsCapture.mainOnMessage!({ data: delta("agent-split", "First half", 0) });
    });
    act(() => {
      // A user message arrives mid-stream, sequenced after the slot's anchor.
      wsCapture.mainOnMessage!({
        data: durable(createUserMessageEvent("mid-stream"), 1),
      });
      wsCapture.mainOnMessage!({
        data: delta("agent-split", " second half", 1),
      });
    });

    // One bubble, still whole, with the interrupting message below it.
    await waitFor(() => {
      const slots = useEventStore
        .getState()
        .uiEvents.filter((event) => isStreamingDeltaEvent(event));
      expect(slots).toHaveLength(1);
      expect(slots[0].content).toBe("First half second half");
    });
    expect(useEventStore.getState().uiEvents.map((event) => event.id)).toEqual([
      "user-msg-conv-split",
      "agent-split",
      "mid-stream",
    ]);
  });

  it("discards buffered deltas from the previous conversation on switch", async () => {
    const { rerender } = renderProviderWithUrl("conv-a");
    await waitFor(() => expect(wsCapture.mainOnMessage).not.toBeNull());
    await waitFor(() => expect(eventIds()).toEqual(["user-msg-conv-a"]));

    // Buffer deltas for A, then switch to B before they flush.
    act(() => {
      wsCapture.mainOnMessage!({ data: itemStarted("stale-item") });
      wsCapture.mainOnMessage!({ data: delta("stale-item", "STALE", 0) });
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

    // B streams and finalizes. Had the switch not reset the batcher, A's
    // "STALE" delta would still be buffered and commit here.
    act(() => {
      wsCapture.mainOnMessage!({ data: itemStarted("agent-b") });
      wsCapture.mainOnMessage!({ data: delta("agent-b", "fresh", 0) });
      wsCapture.mainOnMessage!({
        data: durable(makeAgentMessage("agent-b", "fresh.")),
      });
    });

    const { uiEvents, events } = useEventStore.getState();
    expect(uiEvents).toHaveLength(2);
    expect((uiEvents[1] as MessageEvent).llm_message.content).toEqual([
      { type: "text", text: "fresh." },
    ]);
    expect(JSON.stringify(uiEvents)).not.toContain("STALE");
    expect(JSON.stringify(events)).not.toContain("STALE");
  });

  describe("session socket resume cursor", () => {
    const mainAfterSeq = () =>
      resolveQueryParams(wsCapture.mainOptions).after_seq;

    it("resumes the first connect from the history page, not the whole log", async () => {
      vi.mocked(EventService.getEventCount).mockResolvedValue(42);
      renderProviderWithUrl("conv-cursor");
      await waitFor(() => expect(wsCapture.mainOptions).not.toBeNull());

      // after_seq=-1 would replay history older than the page through
      // non-idempotent side effects (terminal, banners, client tools).
      await waitFor(() => expect(mainAfterSeq()).toBe("41"));
    });

    it("advances past each durable frame, so a reconnect resumes there", async () => {
      renderProviderWithUrl("conv-advance");
      await waitFor(() => expect(wsCapture.mainOnMessage).not.toBeNull());
      // Connect: count is 1, so the socket resumes after seq 0.
      await waitFor(() => expect(mainAfterSeq()).toBe("0"));

      act(() => {
        wsCapture.mainOnMessage!({
          data: durable(makeAgentMessage("agent-1", "hi"), 1),
        });
        wsCapture.mainOnMessage!({
          data: durable(makeAgentMessage("agent-2", "hi"), 2),
        });
      });

      expect(mainAfterSeq()).toBe("2");
    });

    it("does not resume past a durable frame still in flight", async () => {
      renderProviderWithUrl("conv-gap");
      await waitFor(() => expect(wsCapture.mainOnMessage).not.toBeNull());
      await waitFor(() => expect(mainAfterSeq()).toBe("0"));

      // agent-server 1.47.0 delivers a state update (#2) ahead of the user
      // message persisted before it (#1). If the socket drops here, resuming
      // after #2 would lose #1 for good.
      act(() => {
        wsCapture.mainOnMessage!({
          data: durable(makeAgentMessage("state-2", "later"), 2),
        });
      });
      expect(mainAfterSeq()).toBe("0");

      act(() => {
        wsCapture.mainOnMessage!({
          data: durable(makeAgentMessage("msg-1", "earlier"), 1),
        });
        wsCapture.mainOnMessage!({
          data: durable(makeAgentMessage("state-2", "later"), 2),
        });
      });
      expect(mainAfterSeq()).toBe("2");
    });

    it("does not carry a cursor into the next conversation", async () => {
      const { rerender } = renderProviderWithUrl("conv-first");
      await waitFor(() => expect(wsCapture.mainOnMessage).not.toBeNull());
      await waitFor(() => expect(mainAfterSeq()).toBe("0"));
      act(() => {
        wsCapture.mainOnMessage!({
          data: durable(makeAgentMessage("agent-first", "hi"), 1),
        });
      });
      expect(mainAfterSeq()).toBe("1");

      vi.mocked(EventService.getEventCount).mockResolvedValue(3);
      rerender(
        <QueryClientProvider client={queryClient}>
          <ConversationWebSocketProvider
            conversationId="conv-second"
            conversationUrl="http://localhost/api"
          >
            <div />
          </ConversationWebSocketProvider>
        </QueryClientProvider>,
      );

      // Resolving params starts the cursor, exactly as a connect does, so wait
      // until conv-second's socket actually exists (it is gated on history).
      await waitFor(() =>
        expect(wsCapture.calls.at(-1)?.url ?? "").toContain("conv-second"),
      );
      // conv-first's position means nothing in conv-second's log.
      expect(mainAfterSeq()).toBe("2");
    });
  });

  it("does not carry the planner's cursor into a different planning conversation", async () => {
    const planner = (id: string): AppConversation =>
      ({
        id,
        conversation_url: `http://planner.example/api/conversations/${id}`,
        session_api_key: null,
      }) as AppConversation;
    const renderWith = (plannerId: string) => (
      <QueryClientProvider client={queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-plan-switch"
          conversationUrl="http://localhost/api"
          subConversationIds={[plannerId]}
          subConversations={[planner(plannerId)]}
        >
          <div />
        </ConversationWebSocketProvider>
      </QueryClientProvider>
    );
    const plannerAfterSeq = () =>
      resolveQueryParams(wsCapture.planningOptions).after_seq;

    const { rerender } = render(renderWith("planner-a"));
    await waitFor(() => expect(wsCapture.planningOnMessage).not.toBeNull());
    expect(plannerAfterSeq()).toBe("-1");
    act(() => {
      wsCapture.planningOnMessage!({
        data: durable(makeAgentMessage("plan-0", "a"), 0),
      });
    });
    expect(plannerAfterSeq()).toBe("0");

    rerender(renderWith("planner-b"));
    await waitFor(() =>
      expect(
        wsCapture.calls.some(({ url }) => url.endsWith("/planner-b")),
      ).toBe(true),
    );
    expect(plannerAfterSeq()).toBe("-1");
  });

  describe("streaming slots across reconnects", () => {
    const slots = () =>
      useEventStore
        .getState()
        .uiEvents.filter((event) => isStreamingDeltaEvent(event));

    it("leaves no orphaned bubble when the socket drops mid-stream", async () => {
      renderProviderWithUrl("conv-drop");
      await waitFor(() => expect(wsCapture.mainOnMessage).not.toBeNull());
      await waitFor(() => expect(eventIds()).toEqual(["user-msg-conv-drop"]));

      act(() => {
        wsCapture.mainOnMessage!({ data: itemStarted("agent-drop") });
        wsCapture.mainOnMessage!({ data: delta("agent-drop", "half a", 0) });
      });
      await waitFor(() => expect(slots()).toHaveLength(1));

      // Connection lost: progress frames are never replayed, so the slot goes.
      act(() => {
        wsCapture.mainOptions!.onClose!({} as CloseEvent);
      });
      expect(slots()).toHaveLength(0);

      // Reconnect. A stale delta for the dead slot must not resurrect it, and
      // the finished message arrives on the cursor exactly once.
      act(() => {
        wsCapture.mainOptions!.onOpen!({} as Event);
        wsCapture.mainOnMessage!({ data: delta("agent-drop", " sentence", 1) });
        wsCapture.mainOnMessage!({
          data: durable(makeAgentMessage("agent-drop", "half a sentence")),
        });
      });

      const { uiEvents } = useEventStore.getState();
      expect(slots()).toHaveLength(0);
      expect(uiEvents.map((event) => event.id)).toEqual([
        "user-msg-conv-drop",
        "agent-drop",
      ]);
    });

    it("drops the slot on item_aborted", async () => {
      renderProviderWithUrl("conv-abort");
      await waitFor(() => expect(wsCapture.mainOnMessage).not.toBeNull());

      act(() => {
        wsCapture.mainOnMessage!({ data: itemStarted("agent-abort") });
        wsCapture.mainOnMessage!({ data: delta("agent-abort", "partial", 0) });
        wsCapture.mainOnMessage!({
          data: JSON.stringify({
            type: "item_aborted",
            item_id: "agent-abort",
            attempt: 1,
            reason: "cancelled",
          }),
        });
      });

      expect(slots()).toHaveLength(0);
    });

    it("re-streams a retry in the same bubble instead of appending to it", async () => {
      renderProviderWithUrl("conv-retry");
      await waitFor(() => expect(wsCapture.mainOnMessage).not.toBeNull());

      act(() => {
        wsCapture.mainOnMessage!({ data: itemStarted("agent-retry") });
        wsCapture.mainOnMessage!({
          data: delta("agent-retry", "first try", 0),
        });
        wsCapture.mainOnMessage!({
          data: JSON.stringify({
            type: "item_started",
            item_id: "agent-retry",
            attempt: 2,
          }),
        });
        wsCapture.mainOnMessage!({
          data: JSON.stringify({
            type: "delta",
            item_id: "agent-retry",
            attempt: 2,
            order: 0,
            kind: "text",
            content: "second try",
          }),
        });
      });

      await waitFor(() => expect(slots()[0]?.content).toBe("second try"));
      expect(slots()).toHaveLength(1);
    });
  });

  // Frames recorded from a real agent-server 1.47.0 session socket, driven by a
  // word-by-word streaming mock LLM. Large strings are trimmed; nothing else is
  // edited, so these pin the client to what the server actually sends —
  // including durable frames that arrive out of seq order.
  describe("recorded agent-server 1.47.0 transcripts", () => {
    type Frame = {
      type: string;
      seq?: number;
      item_id?: string;
      event?: { id: string; source?: string; llm_message?: unknown };
    };
    const load = <T,>(name: string): T =>
      JSON.parse(
        readFileSync(
          resolve(process.cwd(), "__tests__/fixtures/session-socket", name),
          "utf8",
        ),
      );
    const feed = (frames: Frame[]) =>
      act(() => {
        for (const frame of frames) {
          wsCapture.mainOnMessage!({ data: JSON.stringify(frame) });
        }
      });
    const slots = () =>
      useEventStore
        .getState()
        .uiEvents.filter((event) => isStreamingDeltaEvent(event));
    const retiringIds = (frames: Frame[]) =>
      frames.filter((f) => f.type === "item_started").map((f) => f.item_id!);

    beforeEach(() => {
      // REST history comes from the same server clock as the recording (naive
      // local time), and predates it.
      vi.mocked(EventService.searchEvents).mockImplementation(
        async (conversationId: string) => ({
          items: [
            {
              ...createUserMessageEvent(`user-msg-${conversationId}`),
              timestamp: "2026-09-16T10:00:00.000000",
            },
          ],
          next_page_id: null,
        }),
      );
    });

    it("keeps one whole bubble when a user message lands mid-stream (#15433)", async () => {
      const frames = load<Frame[]>(
        "mid-stream-user-message.agent-server-1.47.0.json",
      );
      renderProviderWithUrl("conv-recorded-a");
      await waitFor(() => expect(wsCapture.mainOnMessage).not.toBeNull());
      await waitFor(() => expect(wsCapture.mainOptions).not.toBeNull());
      resolveQueryParams(wsCapture.mainOptions); // connect

      const midUser = frames.findIndex(
        (f) => f.type === "durable" && f.event?.source === "user" && f.seq! > 1,
      );
      expect(midUser).toBeGreaterThan(
        frames.findIndex((f) => f.type === "item_started"),
      );

      // Up to and including the first delta after the interrupting message.
      feed(frames.slice(0, midUser + 2));
      await waitFor(() =>
        expect(slots()[0]?.content).toBe("Let me check the workspace"),
      );
      expect(slots()).toHaveLength(1);
      const order = useEventStore.getState().uiEvents.map((e) => e.id);
      expect(order.indexOf(slots()[0].id)).toBeLessThan(
        order.indexOf(frames[midUser].event!.id),
      );

      feed(frames.slice(midUser + 2));
      await waitFor(() => expect(slots()).toHaveLength(0));

      const { events } = useEventStore.getState();
      const ids = events.map((e) => e.id);
      for (const itemId of retiringIds(frames)) {
        expect(ids.filter((id) => id === itemId)).toHaveLength(1);
      }
      expect(
        events.filter((e) => e.source === "agent" && "llm_message" in e),
      ).toHaveLength(1);
      // The cursor survived the out-of-order delivery and reached the end.
      const lastSeq = Math.max(
        ...frames.filter((f) => f.type === "durable").map((f) => f.seq!),
      );
      expect(resolveQueryParams(wsCapture.mainOptions).after_seq).toBe(
        String(lastSeq),
      );
    });

    it("leaves no orphan and loses nothing across a real mid-stream reconnect", async () => {
      const { before, resumeAfterSeq, after } = load<{
        before: Frame[];
        resumeAfterSeq: number;
        after: Frame[];
      }>("reconnect-mid-stream.agent-server-1.47.0.json");
      renderProviderWithUrl("conv-recorded-b");
      await waitFor(() => expect(wsCapture.mainOnMessage).not.toBeNull());
      await waitFor(() => expect(wsCapture.mainOptions).not.toBeNull());
      resolveQueryParams(wsCapture.mainOptions); // connect

      feed(before);
      await waitFor(() => expect(slots()).toHaveLength(1));
      const openItem = slots()[0].id;

      // The client computes the same resume point the recording used.
      act(() => {
        wsCapture.mainOptions!.onClose!({} as CloseEvent);
      });
      expect(slots()).toHaveLength(0);
      expect(resolveQueryParams(wsCapture.mainOptions).after_seq).toBe(
        String(resumeAfterSeq),
      );
      act(() => {
        wsCapture.mainOptions!.onOpen!({} as Event);
      });

      feed(after);
      await waitFor(() => {
        expect(slots()).toHaveLength(0);
      });
      const ids = useEventStore.getState().events.map((e) => e.id);
      expect(ids.filter((id) => id === openItem)).toHaveLength(1);
      const serverIds = new Set(
        [...before, ...after]
          .filter((f) => f.type === "durable")
          .map((f) => f.event!.id),
      );
      for (const id of serverIds) {
        expect(ids).toContain(id);
      }
      expect(new Set(ids).size).toBe(ids.length);
    });
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
