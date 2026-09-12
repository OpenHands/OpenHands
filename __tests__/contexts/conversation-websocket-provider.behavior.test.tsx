import React from "react";
import { act, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ConversationWebSocketProvider,
  useConversationWebSocket,
} from "#/contexts/conversation-websocket-context";
import type { WebSocketHookOptions } from "#/hooks/use-websocket";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import { useEventStore } from "#/stores/use-event-store";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { useConversationStateStore } from "#/stores/conversation-state-store";
import { useConversationStore } from "#/stores/conversation-store";
import { useCommandStore } from "#/stores/command-store";
import { useBrowserStore } from "#/stores/browser-store";
import { useGoalStore } from "#/stores/goal-store";
import { useModelStore } from "#/stores/model-store";
import useMetricsStore from "#/stores/metrics-store";
import { useFilesTabStore } from "#/stores/files-tab-store";
import EventService from "#/api/event-service/event-service.api";
import { SERVER_CONNECTION_ERROR_MESSAGE } from "#/constants/server-connection-error";
import {
  getStoredConversationMetadata,
  setStoredConversationMetadata,
} from "#/api/conversation-metadata-store";
import {
  getConversationState,
  setConversationState,
} from "#/utils/conversation-local-storage";

interface TestSocket {
  readonly readyState: number;
  send: WebSocket["send"];
}

const socketCapture = vi.hoisted(() => ({
  callIndex: 0,
  mainUrl: "",
  planningUrl: "",
  mainOptions: null as WebSocketHookOptions | null,
  planningOptions: null as WebSocketHookOptions | null,
  mainSocket: null as TestSocket | null,
  planningSocket: null as TestSocket | null,
  reconnectMain: vi.fn(),
  reconnectPlanning: vi.fn(),
  queueMessage: vi.fn(),
  readConversationFile: vi.fn(),
  trackError: vi.fn(),
}));

const historyCapture = vi.hoisted(() => ({
  result: {
    data: { events: [] as OpenHandsEvent[] } as
      | { events: OpenHandsEvent[] }
      | undefined,
    isPending: false,
    isFetching: false,
    isError: false,
  },
}));

vi.mock("#/hooks/use-websocket", () => ({
  useWebSocket: vi.fn((url: string, options?: WebSocketHookOptions) => {
    const isMain = socketCapture.callIndex % 2 === 0;
    socketCapture.callIndex += 1;

    if (isMain) {
      socketCapture.mainUrl = url;
      socketCapture.mainOptions = options ?? null;
      return {
        socket: socketCapture.mainSocket,
        reconnect: socketCapture.reconnectMain,
      };
    }

    socketCapture.planningUrl = url;
    socketCapture.planningOptions = options ?? null;
    return {
      socket: socketCapture.planningSocket,
      reconnect: socketCapture.reconnectPlanning,
    };
  }),
}));

vi.mock("#/hooks/query/use-conversation-history", () => ({
  useConversationHistory: () => historyCapture.result,
}));

vi.mock("#/hooks/mutation/use-read-conversation-file", () => ({
  useReadConversationFile: () => ({
    mutate: socketCapture.readConversationFile,
  }),
}));

vi.mock("#/utils/error-handler", () => ({
  trackError: (...args: unknown[]) => socketCapture.trackError(...args),
}));

vi.mock("#/api/agent-server-client-options", () => ({
  getAgentServerClientOptions: () => ({ baseUrl: "http://agent-server.test" }),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  ConversationClient: class ConversationClient {
    sendEvent(...args: unknown[]) {
      return socketCapture.queueMessage(...args);
    }
  },
}));

const contextCapture: {
  current: ReturnType<typeof useConversationWebSocket>;
} = { current: null };

function ContextProbe() {
  const context = useConversationWebSocket();
  contextCapture.current = context;

  return (
    <>
      <div data-testid="connection-state">{context?.connectionState}</div>
      <div data-testid="loading-history">
        {String(context?.isLoadingHistory)}
      </div>
    </>
  );
}

type ProviderProps = Omit<
  React.ComponentProps<typeof ConversationWebSocketProvider>,
  "children"
>;

function renderProvider(overrides: Partial<ProviderProps> = {}) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const props: ProviderProps = {
    conversationId: "conv-main",
    conversationUrl: "http://localhost:8000/api/conversations/conv-main",
    ...overrides,
  };

  const view = render(
    <QueryClientProvider client={queryClient}>
      <ConversationWebSocketProvider {...props}>
        <ContextProbe />
      </ConversationWebSocketProvider>
    </QueryClientProvider>,
  );

  return { ...view, queryClient };
}

function makeSocket(readyState: number = WebSocket.OPEN): TestSocket {
  return {
    readyState,
    send: vi.fn(),
  };
}

const makeSendRequest = (text: string) => ({
  role: "user" as const,
  content: [{ type: "text" as const, text }],
});

function makeSubConversation(
  overrides: Partial<AppConversation> = {},
): AppConversation {
  return {
    id: "conv-planning",
    created_by_user_id: null,
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    title: "Planning",
    trigger: null,
    pr_number: [],
    llm_model: null,
    metrics: null,
    created_at: "2024-01-01T00:00:00.000Z",
    updated_at: "2024-01-01T00:00:00.000Z",
    execution_status: null,
    conversation_url: "http://localhost:8000/api/conversations/conv-planning",
    session_api_key: "planning-key",
    sandbox_id: null,
    sub_conversation_ids: [],
    ...overrides,
  };
}

const baseEvent = (
  id: string,
  source: "agent" | "user" | "environment" = "environment",
) => ({
  id,
  timestamp: `2024-01-01T00:00:${id.padStart(2, "0")}.000Z`,
  source,
});

function makeMessageEvent(
  id: string,
  role: "user" | "assistant",
  textParts: string[],
): OpenHandsEvent {
  return {
    ...baseEvent(id, role === "user" ? "user" : "agent"),
    llm_message: {
      role,
      content: textParts.flatMap((text, index) =>
        index === 0
          ? [
              { type: "image" as const, image_url: "data:image/png;base64,AA" },
              { type: "text" as const, text },
            ]
          : [{ type: "text" as const, text }],
      ),
    },
    activated_microagents: [],
    extended_content: [],
  } as OpenHandsEvent;
}

function makeActionEvent(
  id: string,
  action: Record<string, unknown>,
  toolName = "execute_bash",
): OpenHandsEvent {
  return {
    ...baseEvent(id, "agent"),
    thought: [],
    thinking_blocks: [],
    action,
    tool_name: toolName,
    tool_call_id: `call-${id}`,
    tool_call: {
      id: `call-${id}`,
      type: "function",
      function: { name: toolName, arguments: "{}" },
    },
    llm_response_id: `response-${id}`,
    security_risk: "UNKNOWN",
  } as OpenHandsEvent;
}

function makeObservationEvent(
  id: string,
  observation: Record<string, unknown>,
  toolName = "execute_bash",
): OpenHandsEvent {
  return {
    ...baseEvent(id),
    action_id: `action-${id}`,
    tool_name: toolName,
    tool_call_id: `call-${id}`,
    observation,
  } as OpenHandsEvent;
}

function makeStateEvent(
  id: string,
  key: "full_state" | "execution_status" | "stats" | "goal",
  value: unknown,
): OpenHandsEvent {
  return {
    ...baseEvent(id),
    kind: "ConversationStateUpdateEvent",
    key,
    value,
  } as OpenHandsEvent;
}

function mainOptions(): WebSocketHookOptions {
  expect(socketCapture.mainOptions).not.toBeNull();
  return socketCapture.mainOptions!;
}

function planningOptions(): WebSocketHookOptions {
  expect(socketCapture.planningOptions).not.toBeNull();
  return socketCapture.planningOptions!;
}

function dispatchMain(event: unknown) {
  act(() => {
    mainOptions().onMessage?.({ data: JSON.stringify(event) } as MessageEvent);
  });
}

function dispatchPlanning(event: unknown) {
  act(() => {
    planningOptions().onMessage?.({
      data: JSON.stringify(event),
    } as MessageEvent);
  });
}

describe("Conversation websocket behavior", () => {
  beforeEach(() => {
    socketCapture.callIndex = 0;
    socketCapture.mainUrl = "";
    socketCapture.planningUrl = "";
    socketCapture.mainOptions = null;
    socketCapture.planningOptions = null;
    socketCapture.mainSocket = null;
    socketCapture.planningSocket = null;
    socketCapture.reconnectMain.mockReset();
    socketCapture.reconnectPlanning.mockReset();
    socketCapture.queueMessage.mockReset().mockResolvedValue(undefined);
    socketCapture.readConversationFile.mockReset();
    socketCapture.trackError.mockReset();
    historyCapture.result = {
      data: { events: [] },
      isPending: false,
      isFetching: false,
      isError: false,
    };
    contextCapture.current = null;

    window.localStorage.clear();
    useEventStore.setState({
      events: [],
      eventIds: new Set(),
      uiEvents: [],
      loadedConversationId: null,
    });
    useErrorMessageStore.getState().removeErrorMessage();
    useOptimisticUserMessageStore.setState({ pendingMessages: [] });
    useConversationStateStore.getState().reset();
    useConversationStore.setState({
      conversationMode: "code",
      planContent: null,
    });
    useCommandStore.getState().clearTerminal();
    useBrowserStore.getState().reset();
    useGoalStore.setState({ statusByConversation: {} });
    useModelStore.getState().clearAll();
    useMetricsStore.setState({
      cost: null,
      max_budget_per_task: null,
      usage: null,
    });
    useFilesTabStore.setState({
      selectedPath: null,
      selectedConversationId: null,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
  });

  it("returns no websocket API outside the provider", () => {
    render(<ContextProbe />);

    expect(contextCapture.current).toBeNull();
  });

  it("hydrates history, consumes matching optimistic messages, and subscribes after the latest event", async () => {
    const first = makeMessageEvent("01", "assistant", ["first"]);
    const latest = makeMessageEvent("02", "user", ["hello", " world"]);
    historyCapture.result.data = { events: [first, latest] };
    const consumeMatchingPendingMessage = vi.spyOn(
      useOptimisticUserMessageStore.getState(),
      "consumeMatchingPendingMessage",
    );
    useOptimisticUserMessageStore.setState({
      pendingMessages: [
        {
          id: "pending-decoy",
          conversationId: "conv-main",
          text: "different message",
          content: "different message",
          status: "sending",
          imageUrls: [],
          fileUrls: [],
          timestamp: "2024-01-01T00:00:00.000Z",
        },
        {
          id: "pending-1",
          conversationId: "conv-main",
          text: "hello world",
          content: "hello world",
          status: "sending",
          imageUrls: [],
          fileUrls: [],
          timestamp: "2024-01-01T00:00:00.000Z",
        },
      ],
    });

    renderProvider({ sessionApiKey: "session-key" });

    await waitFor(() =>
      expect(useEventStore.getState().events).toHaveLength(2),
    );
    expect(useOptimisticUserMessageStore.getState().pendingMessages).toEqual([
      expect.objectContaining({ id: "pending-decoy" }),
    ]);
    expect(consumeMatchingPendingMessage).toHaveBeenCalledOnce();
    expect(consumeMatchingPendingMessage).toHaveBeenCalledWith(
      "conv-main",
      "hello world",
    );
    expect(socketCapture.mainUrl).toContain("/sockets/events/conv-main");
    // The session key is now passed via the dedicated `sessionApiKey` option
    // rather than folded into the query string.
    expect(mainOptions().queryParams).toEqual({
      resend_mode: "since",
      after_timestamp: latest.timestamp,
    });
    expect(mainOptions().sessionApiKey).toBe("session-key");
    expect(planningOptions().queryParams).toEqual({
      resend_all: true,
    });
    expect(planningOptions().sessionApiKey).toBe("session-key");
    expect(mainOptions().reconnect).toEqual({ enabled: true });
    expect(planningOptions().reconnect).toEqual({ enabled: true });
  });

  it("waits for the first history load before opening the main socket", () => {
    // The socket gate keys on `isPending` (the genuine first load with no
    // cached data) rather than `isFetching`, so background refetches never
    // tear down a live socket. During that first load the URL stays blank.
    historyCapture.result.data = undefined;
    historyCapture.result.isPending = true;

    renderProvider();

    expect(socketCapture.mainUrl).toBe("");
    expect(mainOptions().queryParams).toEqual({ resend_mode: "all" });
    expect(screen.getByTestId("connection-state")).toHaveTextContent(
      "CONNECTING",
    );
  });

  it("shows first-load history only for a selected conversation", () => {
    historyCapture.result.isPending = true;

    const selected = renderProvider();
    expect(screen.getByTestId("loading-history")).toHaveTextContent("true");
    selected.unmount();

    socketCapture.callIndex = 0;
    renderProvider({ conversationId: undefined });
    expect(screen.getByTestId("loading-history")).toHaveTextContent("false");
  });

  it("falls back to an all-events socket when history loading fails", () => {
    historyCapture.result.isFetching = true;
    historyCapture.result.isError = true;

    renderProvider();

    expect(socketCapture.mainUrl).toContain("/sockets/events/conv-main");
    expect(mainOptions().queryParams).toEqual({ resend_mode: "all" });
  });

  it("does not build socket URLs without complete conversation coordinates", () => {
    const missingBoth = renderProvider({
      conversationId: undefined,
      conversationUrl: null,
    });

    expect(socketCapture.mainUrl).toBe("");
    expect(socketCapture.planningUrl).toBe("");
    expect(useEventStore.getState().loadedConversationId).toBeNull();
    missingBoth.unmount();

    socketCapture.callIndex = 0;
    const missingUrl = renderProvider({ conversationUrl: null });
    expect(socketCapture.mainUrl).toBe("");
    missingUrl.unmount();

    socketCapture.callIndex = 0;
    renderProvider({
      conversationId: undefined,
      conversationUrl: "http://localhost:8000/api/conversations/conv-main",
    });
    expect(socketCapture.mainUrl).toBe("");
  });

  it("accepts absent history and history without a selected conversation", () => {
    const addEvents = vi.spyOn(useEventStore.getState(), "addEvents");
    const empty = renderProvider();
    expect(addEvents).not.toHaveBeenCalled();
    empty.unmount();

    socketCapture.callIndex = 0;
    historyCapture.result.data = undefined;
    const first = renderProvider();
    expect(mainOptions().queryParams).toEqual({ resend_mode: "all" });
    expect(addEvents).not.toHaveBeenCalled();
    first.unmount();

    socketCapture.callIndex = 0;
    const consumeMatchingPendingMessage = vi.spyOn(
      useOptimisticUserMessageStore.getState(),
      "consumeMatchingPendingMessage",
    );
    historyCapture.result.data = {
      events: [makeMessageEvent("04", "user", ["unscoped history"])],
    };
    renderProvider({ conversationId: undefined });
    expect(useEventStore.getState().events).toHaveLength(1);
    expect(consumeMatchingPendingMessage).not.toHaveBeenCalled();
    expect(useOptimisticUserMessageStore.getState().pendingMessages).toEqual(
      [],
    );
  });

  it("clears conversation-owned stores once per conversation identity", () => {
    useEventStore
      .getState()
      .addEvent(makeMessageEvent("05", "assistant", ["old conversation"]));
    useEventStore.setState({ loadedConversationId: "conv-old" });
    useBrowserStore.getState().setUrl("https://old.example");

    const first = renderProvider();
    expect(useEventStore.getState()).toMatchObject({
      events: [],
      loadedConversationId: "conv-main",
    });
    expect(useBrowserStore.getState().url).toBe("");
    first.unmount();

    useEventStore
      .getState()
      .addEvent(makeMessageEvent("06", "assistant", ["keep on remount"]));
    useBrowserStore.getState().setUrl("https://current.example");
    socketCapture.callIndex = 0;
    const sameConversation = renderProvider();
    expect(useEventStore.getState().events).toHaveLength(1);
    expect(useBrowserStore.getState().url).toBe("https://current.example");

    sameConversation.rerender(
      <QueryClientProvider client={sameConversation.queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-next"
          conversationUrl="http://localhost:8000/api/conversations/conv-next"
        >
          <ContextProbe />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );
    expect(useEventStore.getState()).toMatchObject({
      events: [],
      loadedConversationId: "conv-next",
    });
    expect(useBrowserStore.getState().url).toBe("");
  });

  it("hydrates and opens the socket when an in-flight history query settles", async () => {
    historyCapture.result.data = undefined;
    historyCapture.result.isPending = true;
    const view = renderProvider();
    expect(socketCapture.mainUrl).toBe("");

    const latest = makeMessageEvent("07", "assistant", ["fresh tail"]);
    historyCapture.result = {
      data: { events: [latest] },
      isPending: false,
      isFetching: false,
      isError: false,
    };
    view.rerender(
      <QueryClientProvider client={view.queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-main"
          conversationUrl="http://localhost:8000/api/conversations/conv-main"
        >
          <ContextProbe />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );

    await waitFor(() =>
      expect(useEventStore.getState().events).toHaveLength(1),
    );
    expect(socketCapture.mainUrl).toContain("/sockets/events/conv-main");
    expect(mainOptions().queryParams).toEqual({
      resend_mode: "since",
      after_timestamp: latest.timestamp,
    });
    expect(
      Object.hasOwn(mainOptions().queryParams ?? {}, "session_api_key"),
    ).toBe(false);
    expect(
      Object.hasOwn(planningOptions().queryParams ?? {}, "session_api_key"),
    ).toBe(false);
  });

  it("ignores incomplete planning conversations and connects complete ones", () => {
    const { rerender, queryClient } = renderProvider({
      subConversations: [makeSubConversation({ conversation_url: null })],
    });
    expect(socketCapture.planningUrl).toBe("");

    rerender(
      <QueryClientProvider client={queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-main"
          conversationUrl="http://localhost:8000/api/conversations/conv-main"
          subConversations={[makeSubConversation()]}
        >
          <ContextProbe />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );

    expect(socketCapture.planningUrl).toContain(
      "/sockets/events/conv-planning",
    );
  });

  it("safely ignores an empty planning conversation slot", () => {
    expect(() =>
      renderProvider({
        subConversations: [undefined as unknown as AppConversation],
      }),
    ).not.toThrow();
    expect(socketCapture.planningUrl).toBe("");
  });

  it("reports the main socket lifecycle and only surfaces errors after a successful connection", () => {
    renderProvider();

    act(() => mainOptions().onError?.(new Event("error")));
    expect(useErrorMessageStore.getState().errorMessage).toBeNull();
    expect(screen.getByTestId("connection-state")).toHaveTextContent("CLOSED");

    useErrorMessageStore
      .getState()
      .setErrorMessage("old connection error", "connection");
    act(() => mainOptions().onOpen?.(new Event("open")));
    expect(screen.getByTestId("connection-state")).toHaveTextContent("OPEN");
    expect(useErrorMessageStore.getState().errorMessage).toBeNull();

    act(() => mainOptions().onClose?.(new CloseEvent("close")));
    expect(screen.getByTestId("connection-state")).toHaveTextContent("CLOSED");

    act(() => mainOptions().onError?.(new Event("error")));
    expect(useErrorMessageStore.getState()).toMatchObject({
      errorMessage: SERVER_CONNECTION_ERROR_MESSAGE,
      errorType: "connection",
    });
  });

  it("merges main and planning connection lifecycles", async () => {
    vi.spyOn(EventService, "getEventCount").mockResolvedValue(0);
    renderProvider({
      subConversations: [makeSubConversation()],
      subConversationIds: ["conv-planning"],
    });
    expect(screen.getByTestId("connection-state")).toHaveTextContent(
      "CONNECTING",
    );
    expect(screen.getByTestId("loading-history")).toHaveTextContent("true");

    await act(async () => {
      mainOptions().onOpen?.(new Event("open"));
      await planningOptions().onOpen?.(new Event("open"));
    });
    expect(screen.getByTestId("connection-state")).toHaveTextContent("OPEN");
    expect(screen.getByTestId("loading-history")).toHaveTextContent("false");

    act(() => planningOptions().onClose?.(new CloseEvent("close")));
    expect(screen.getByTestId("connection-state")).toHaveTextContent("CLOSED");

    act(() => mainOptions().onClose?.(new CloseEvent("close")));
    expect(screen.getByTestId("connection-state")).toHaveTextContent("CLOSED");

    act(() => planningOptions().onError?.(new Event("error")));
    expect(useErrorMessageStore.getState()).toMatchObject({
      errorMessage: SERVER_CONNECTION_ERROR_MESSAGE,
      errorType: "connection",
    });
  });

  for (const [label, readyState, expected] of [
    ["connecting", WebSocket.CONNECTING, "CONNECTING"],
    ["open", WebSocket.OPEN, "OPEN"],
    ["closing", WebSocket.CLOSING, "CLOSING"],
    ["closed", WebSocket.CLOSED, "CLOSED"],
    ["unknown", 99, "CLOSED"],
  ] as const) {
    it(`reflects a ${label} main socket ready state`, async () => {
      socketCapture.mainSocket = makeSocket(readyState);
      renderProvider();

      await waitFor(() =>
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          expected,
        ),
      );
    });
  }

  it("uses planning socket ready state when both conversations are connected", async () => {
    socketCapture.mainSocket = makeSocket(WebSocket.OPEN);
    socketCapture.planningSocket = makeSocket(WebSocket.CLOSING);
    renderProvider({ subConversations: [makeSubConversation()] });

    await waitFor(() =>
      expect(screen.getByTestId("connection-state")).toHaveTextContent(
        "CLOSING",
      ),
    );
  });

  it("keeps the planning connection in its initial connecting state", async () => {
    socketCapture.mainSocket = makeSocket(WebSocket.OPEN);
    renderProvider({ subConversations: [makeSubConversation()] });

    await waitFor(() =>
      expect(screen.getByTestId("connection-state")).toHaveTextContent(
        "CONNECTING",
      ),
    );
  });

  for (const [label, mainReadyState, planningReadyState, expected] of [
    ["main connecting", WebSocket.CONNECTING, WebSocket.OPEN, "CONNECTING"],
    ["main closed", WebSocket.CLOSED, WebSocket.OPEN, "CLOSED"],
    ["planning closing", WebSocket.CLOSED, WebSocket.CLOSING, "CLOSING"],
    ["main closing", WebSocket.CLOSING, WebSocket.CLOSED, "CLOSING"],
  ] as const) {
    it(`merges ${label} with the other socket`, async () => {
      socketCapture.mainSocket = makeSocket(mainReadyState);
      socketCapture.planningSocket = makeSocket(planningReadyState);
      renderProvider({ subConversations: [makeSubConversation()] });

      await waitFor(() =>
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          expected,
        ),
      );
    });
  }

  for (const [label, readyState, expected] of [
    ["connecting", WebSocket.CONNECTING, "CONNECTING"],
    ["closed", WebSocket.CLOSED, "CLOSED"],
    ["unknown", 99, "CLOSED"],
  ] as const) {
    it(`reflects a ${label} planning socket ready state`, async () => {
      socketCapture.mainSocket = makeSocket(WebSocket.OPEN);
      socketCapture.planningSocket = makeSocket(readyState);
      renderProvider({ subConversations: [makeSubConversation()] });

      await waitFor(() =>
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          expected,
        ),
      );
    });
  }

  it("resets connection history and planning loading when identities change", () => {
    const view = renderProvider({ subConversations: [makeSubConversation()] });
    act(() => mainOptions().onOpen?.(new Event("open")));

    view.rerender(
      <QueryClientProvider client={view.queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-main"
          conversationUrl="http://localhost:8000/api/conversations/conv-main"
          subConversations={[makeSubConversation()]}
          subConversationIds={["conv-planning"]}
        >
          <ContextProbe />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );
    expect(screen.getByTestId("loading-history")).toHaveTextContent("true");
    act(() => mainOptions().onError?.(new Event("error")));
    expect(useErrorMessageStore.getState().errorMessage).toBeNull();

    act(() => mainOptions().onOpen?.(new Event("open")));
    view.rerender(
      <QueryClientProvider client={view.queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-next"
          conversationUrl="http://localhost:8000/api/conversations/conv-next"
          subConversations={[makeSubConversation()]}
          subConversationIds={["conv-planning"]}
        >
          <ContextProbe />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );
    act(() => mainOptions().onError?.(new Event("error")));
    expect(useErrorMessageStore.getState().errorMessage).toBeNull();
  });

  it("forgets a successful main connection when the conversation changes", () => {
    const view = renderProvider();
    act(() => mainOptions().onOpen?.(new Event("open")));

    view.rerender(
      <QueryClientProvider client={view.queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-next"
          conversationUrl="http://localhost:8000/api/conversations/conv-next"
        >
          <ContextProbe />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );
    act(() => mainOptions().onError?.(new Event("error")));

    expect(useErrorMessageStore.getState().errorMessage).toBeNull();
  });

  it("dismisses the error and reconnects the socket for the active mode", () => {
    const { unmount } = renderProvider({
      subConversations: [makeSubConversation()],
    });
    useErrorMessageStore.getState().setErrorMessage("retry me", "connection");
    useConversationStore.setState({ conversationMode: "plan" });

    act(() => contextCapture.current?.reconnect());

    expect(useErrorMessageStore.getState().errorMessage).toBeNull();
    expect(socketCapture.reconnectPlanning).toHaveBeenCalledOnce();
    expect(socketCapture.reconnectMain).not.toHaveBeenCalled();

    useConversationStore.setState({ conversationMode: "code" });
    act(() => contextCapture.current?.reconnect());
    expect(socketCapture.reconnectMain).toHaveBeenCalledOnce();
    expect(socketCapture.reconnectPlanning).toHaveBeenCalledOnce();

    unmount();
    socketCapture.callIndex = 0;
    socketCapture.reconnectMain.mockClear();
    renderProvider();
    useConversationStore.setState({ conversationMode: "plan" });
    act(() => contextCapture.current?.reconnect());
    expect(socketCapture.reconnectMain).toHaveBeenCalledOnce();
  });

  it("uses sockets and reconnect routing introduced after the first render", async () => {
    const view = renderProvider();
    const mainSocket = makeSocket(WebSocket.OPEN);
    const planningSocket = makeSocket(WebSocket.OPEN);
    socketCapture.mainSocket = mainSocket;
    socketCapture.planningSocket = planningSocket;

    view.rerender(
      <QueryClientProvider client={view.queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-main"
          conversationUrl="http://localhost:8000/api/conversations/conv-main"
          subConversations={[makeSubConversation()]}
        >
          <ContextProbe />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );

    await waitFor(() =>
      expect(screen.getByTestId("connection-state")).toHaveTextContent("OPEN"),
    );
    await expect(
      contextCapture.current?.sendMessage(makeSendRequest("new socket")),
    ).resolves.toEqual({ queued: false });
    expect(mainSocket.send).toHaveBeenCalledWith(
      JSON.stringify({ ...makeSendRequest("new socket"), run: true }),
    );
    expect(socketCapture.queueMessage).not.toHaveBeenCalled();

    useConversationStore.setState({ conversationMode: "plan" });
    act(() => contextCapture.current?.reconnect());
    expect(socketCapture.reconnectPlanning).toHaveBeenCalledOnce();
    expect(socketCapture.reconnectMain).not.toHaveBeenCalled();
  });

  it("sends code and plan messages through their open sockets", async () => {
    const mainSocket = makeSocket();
    const planningSocket = makeSocket();
    socketCapture.mainSocket = mainSocket;
    socketCapture.planningSocket = planningSocket;
    renderProvider({ subConversations: [makeSubConversation()] });

    await expect(
      contextCapture.current?.sendMessage(makeSendRequest("code message")),
    ).resolves.toEqual({ queued: false });
    expect(mainSocket.send).toHaveBeenCalledWith(
      JSON.stringify({ ...makeSendRequest("code message"), run: true }),
    );

    useConversationStore.setState({ conversationMode: "plan" });
    await expect(
      contextCapture.current?.sendMessage(makeSendRequest("plan message")),
    ).resolves.toEqual({ queued: false });
    expect(planningSocket.send).toHaveBeenCalledWith(
      JSON.stringify({ ...makeSendRequest("plan message"), run: true }),
    );
  });

  it("queues messages over REST while the socket is unavailable", async () => {
    renderProvider();

    await expect(
      contextCapture.current?.sendMessage(makeSendRequest("queue me")),
    ).resolves.toEqual({ queued: true });
    expect(socketCapture.queueMessage).toHaveBeenCalledWith(
      "conv-main",
      { role: "user", content: makeSendRequest("queue me").content },
      { run: true },
    );
  });

  it("reports missing conversations and REST queue failures", async () => {
    renderProvider({ conversationId: undefined });

    await expect(
      contextCapture.current?.sendMessage(makeSendRequest("orphan")),
    ).rejects.toThrow("No conversation ID available");
    expect(useErrorMessageStore.getState().errorMessage).toBe(
      "No conversation ID available",
    );

    socketCapture.callIndex = 0;
    socketCapture.queueMessage.mockRejectedValueOnce(new Error("queue broke"));
    renderProvider();
    await expect(
      contextCapture.current?.sendMessage(makeSendRequest("retry")),
    ).rejects.toThrow("queue broke");
    expect(useErrorMessageStore.getState().errorMessage).toBe("queue broke");
  });

  it("uses a safe error for non-Error queue failures", async () => {
    socketCapture.queueMessage.mockRejectedValueOnce("offline");
    renderProvider();

    await expect(
      contextCapture.current?.sendMessage(makeSendRequest("retry")),
    ).rejects.toBe("offline");
    expect(useErrorMessageStore.getState().errorMessage).toBe(
      "Failed to queue message for delivery",
    );
  });

  it("reports WebSocket send failures", async () => {
    const socket = makeSocket();
    vi.mocked(socket.send).mockImplementationOnce(() => {
      throw new Error("send broke");
    });
    socketCapture.mainSocket = socket;
    renderProvider();

    await expect(
      contextCapture.current?.sendMessage(makeSendRequest("hello")),
    ).rejects.toThrow("send broke");
    expect(useErrorMessageStore.getState().errorMessage).toBe("send broke");

    vi.mocked(socket.send).mockImplementationOnce(() => {
      throw "closed";
    });
    await expect(
      contextCapture.current?.sendMessage(makeSendRequest("again")),
    ).rejects.toBe("closed");
    expect(useErrorMessageStore.getState().errorMessage).toBe(
      "Failed to send message",
    );
  });

  it("routes main-conversation events to their user-visible stores", () => {
    const consumeMatchingPendingMessage = vi.spyOn(
      useOptimisticUserMessageStore.getState(),
      "consumeMatchingPendingMessage",
    );
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const { queryClient } = renderProvider();
    const invalidateQueries = vi.spyOn(queryClient, "invalidateQueries");

    useErrorMessageStore.getState().setErrorMessage("temporary", "connection");
    dispatchMain(makeMessageEvent("10", "assistant", ["connected"]));
    expect(useErrorMessageStore.getState().errorMessage).toBeNull();
    expect(consumeMatchingPendingMessage).not.toHaveBeenCalled();
    expect(socketCapture.trackError).not.toHaveBeenCalled();
    expect(warn).not.toHaveBeenCalled();

    dispatchMain({
      ...baseEvent("11"),
      kind: "ConversationErrorEvent",
      code: "BadRequest",
      detail: "Conversation failed",
    });
    expect(useErrorMessageStore.getState()).toMatchObject({
      errorMessage: "Conversation failed",
      errorType: "conversation",
      errorCode: "BadRequest",
    });
    expect(socketCapture.trackError).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "conversation",
        metadata: expect.objectContaining({
          eventId: "11",
          errorCode: "BadRequest",
        }),
      }),
    );
    expect(socketCapture.trackError).toHaveBeenCalledTimes(1);

    dispatchMain(makeMessageEvent("12", "assistant", ["still connected"]));
    expect(useErrorMessageStore.getState().errorMessage).toBe(
      "Conversation failed",
    );

    dispatchMain({
      ...baseEvent("13", "agent"),
      tool_name: "llm",
      tool_call_id: "call-13",
      error: "Model failed",
    });
    expect(socketCapture.trackError).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "agent",
        metadata: expect.objectContaining({
          eventId: "13",
          toolName: "llm",
          toolCallId: "call-13",
        }),
      }),
    );
    expect(socketCapture.trackError).toHaveBeenCalledTimes(2);

    useOptimisticUserMessageStore.setState({
      pendingMessages: [
        {
          id: "pending-main",
          conversationId: "conv-main",
          text: "hello world",
          content: "hello world",
          status: "sending",
          imageUrls: [],
          fileUrls: [],
          timestamp: "2024-01-01T00:00:00.000Z",
        },
      ],
    });
    setConversationState("conv-main", { draftMessage: "hello world" });
    dispatchMain(makeMessageEvent("14", "user", ["hello", " world"]));
    expect(useOptimisticUserMessageStore.getState().pendingMessages).toEqual(
      [],
    );
    expect(consumeMatchingPendingMessage).toHaveBeenCalledOnce();
    expect(consumeMatchingPendingMessage).toHaveBeenCalledWith(
      "conv-main",
      "hello world",
    );
    expect(getConversationState("conv-main").draftMessage).toBeNull();

    dispatchMain(
      makeActionEvent("15", {
        kind: "ExecuteBashAction",
        command: "pwd",
        is_input: false,
        timeout: null,
        reset: false,
      }),
    );
    expect(useCommandStore.getState().commands).toContainEqual({
      type: "input",
      content: "pwd",
    });
    expect(invalidateQueries).toHaveBeenCalledWith(
      { queryKey: ["file_changes", "conv-main"] },
      { cancelRefetch: false },
    );

    dispatchMain(
      makeStateEvent("16", "full_state", {
        execution_status: "running",
      }),
    );
    // Execution status is now scoped per conversation id.
    expect(
      useConversationStateStore.getState().executionStatusByConversation[
        "conv-main"
      ],
    ).toBe("running");
    dispatchMain(makeStateEvent("17", "execution_status", "paused"));
    expect(
      useConversationStateStore.getState().executionStatusByConversation[
        "conv-main"
      ],
    ).toBe("paused");

    dispatchMain(
      makeStateEvent("18-no-metrics", "stats", {
        usage_to_metrics: undefined,
      }),
    );
    dispatchMain(makeStateEvent("18", "stats", { usage_to_metrics: {} }));
    expect(warn).not.toHaveBeenCalled();
    dispatchMain(
      makeStateEvent("19", "stats", {
        usage_to_metrics: {
          agent: {
            accumulated_cost: 1.25,
            max_budget_per_task: undefined,
            accumulated_token_usage: null,
          },
        },
      }),
    );
    expect(useMetricsStore.getState()).toMatchObject({
      cost: 1.25,
      usage: null,
    });
    // A stats update whose budget is absent passes the value through as
    // undefined rather than coercing it to null.
    expect(useMetricsStore.getState().max_budget_per_task).toBeUndefined();
    dispatchMain(
      makeStateEvent("20", "stats", {
        usage_to_metrics: {
          agent: {
            accumulated_cost: 2.5,
            max_budget_per_task: 8,
            accumulated_token_usage: {
              prompt_tokens: 10,
              completion_tokens: 20,
              cache_read_tokens: 3,
              cache_write_tokens: 4,
              context_window: 128000,
              per_turn_token: 30,
            },
          },
        },
      }),
    );
    expect(useMetricsStore.getState()).toMatchObject({
      cost: 2.5,
      max_budget_per_task: 8,
      usage: {
        prompt_tokens: 10,
        completion_tokens: 20,
        cache_read_tokens: 3,
        cache_write_tokens: 4,
        context_window: 128000,
        per_turn_token: 30,
      },
    });
    expect(
      useConversationStateStore.getState().executionStatusByConversation[
        "conv-main"
      ],
    ).toBe("paused");

    const goalStatus = {
      active: false,
      status: "complete",
      iteration: 2,
      max_iterations: 3,
      objective: "Ship it",
      verdict: { score: 1, complete: true, missing: "" },
    };
    dispatchMain(makeStateEvent("21", "goal", goalStatus));
    expect(useGoalStore.getState().statusByConversation["conv-main"]).toEqual(
      goalStatus,
    );
    expect(
      useConversationStateStore.getState().executionStatusByConversation[
        "conv-main"
      ],
    ).toBe("paused");

    dispatchMain(
      makeObservationEvent("22", {
        kind: "ExecuteBashObservation",
        content: [
          { type: "text", text: "line one" },
          { type: "image", image_urls: ["ignored"] },
          { type: "text", text: "line two" },
        ],
      }),
    );
    expect(useCommandStore.getState().commands).toContainEqual({
      type: "output",
      content: "line one\nline two",
    });

    dispatchMain(
      makeObservationEvent(
        "23",
        { kind: "BrowserObservation", screenshot_data: "raw-image" },
        "browser",
      ),
    );
    expect(useBrowserStore.getState().screenshotSrc).toBe(
      "data:image/png;base64,raw-image",
    );
    dispatchMain(
      makeObservationEvent(
        "24",
        {
          kind: "BrowserObservation",
          screenshot_data: "data:image/webp;base64,ready",
        },
        "browser",
      ),
    );
    expect(useBrowserStore.getState().screenshotSrc).toBe(
      "data:image/webp;base64,ready",
    );
    dispatchMain(
      makeObservationEvent(
        "25",
        { kind: "BrowserObservation", screenshot_data: null },
        "browser",
      ),
    );

    dispatchMain(
      makeActionEvent(
        "26",
        { kind: "BrowserNavigateAction", url: "https://example.com" },
        "browser",
      ),
    );
    expect(useBrowserStore.getState().url).toBe("https://example.com");

    queryClient.setQueryData(["user", "conversation", "conv-main"], {
      id: "conv-main",
      llm_model: "old-model",
    });
    dispatchMain(
      makeObservationEvent(
        "27",
        {
          kind: "SwitchLLMObservation",
          content: [{ type: "text", text: "Switched" }],
          is_error: false,
          profile_name: "fast-profile",
          reason: null,
          active_model: "new-model",
        },
        "switch_llm",
      ),
    );
    expect(
      useModelStore.getState().activeProfileByConversation["conv-main"],
    ).toBe("fast-profile");
    // The live stamp merges only the profile fields over the default
    // repo/branch/provider scaffold; optional fields it doesn't own
    // (selected_workspace, plugins) are left unset rather than nulled.
    expect(getStoredConversationMetadata("conv-main")).toMatchObject({
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
      active_profile: "fast-profile",
    });
    expect(
      queryClient.getQueryData(["user", "conversation", "conv-main"]),
    ).toMatchObject({ llm_model: "new-model" });

    dispatchMain(
      makeObservationEvent(
        "28",
        {
          kind: "SwitchLLMObservation",
          content: [],
          is_error: true,
          profile_name: "failed-profile",
          reason: "unavailable",
          active_model: null,
        },
        "switch_llm",
      ),
    );
    expect(
      useModelStore.getState().activeProfileByConversation["conv-main"],
    ).toBe("fast-profile");

    dispatchMain(
      makeObservationEvent(
        "28-without-model",
        {
          kind: "SwitchLLMObservation",
          content: [{ type: "text", text: "Profile switched" }],
          is_error: false,
          profile_name: "profile-without-model",
          reason: null,
          active_model: null,
        },
        "switch_llm",
      ),
    );
    expect(
      useModelStore.getState().activeProfileByConversation["conv-main"],
    ).toBe("profile-without-model");
    expect(getStoredConversationMetadata("conv-main")).toMatchObject({
      active_profile: "profile-without-model",
    });
    expect(
      queryClient.getQueryData(["user", "conversation", "conv-main"]),
    ).toMatchObject({ llm_model: "new-model" });

    dispatchMain(
      makeActionEvent(
        "29",
        {
          kind: "CanvasUIAction",
          command: "open_tab",
          tab: "terminal",
        },
        "canvas_ui",
      ),
    );
    expect(useConversationStore.getState()).toMatchObject({
      selectedTab: "terminal",
      isRightPanelShown: true,
    });
    expect(warn).not.toHaveBeenCalled();
  });

  it("uses the latest conversation identity for canvas file navigation", () => {
    const view = renderProvider();
    view.rerender(
      <QueryClientProvider client={view.queryClient}>
        <ConversationWebSocketProvider
          conversationId="conv-next"
          conversationUrl="http://localhost:8000/api/conversations/conv-next"
        >
          <ContextProbe />
        </ConversationWebSocketProvider>
      </QueryClientProvider>,
    );

    dispatchMain(
      makeActionEvent(
        "30",
        {
          kind: "CanvasUIAction",
          command: "navigate_to_file",
          path: "/workspace/src/index.ts",
        },
        "canvas_ui",
      ),
    );
    // The canvas navigation handler normalizes the agent path to a
    // workspace-relative one, stripping the `/workspace/<root>/` prefix.
    expect(useFilesTabStore.getState()).toMatchObject({
      selectedPath: "index.ts",
      selectedConversationId: "conv-next",
    });
  });

  it("ignores malformed and non-event messages without disrupting the socket", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    renderProvider();

    act(() => mainOptions().onMessage?.({ data: "not-json" } as MessageEvent));
    dispatchMain({ type: "not-an-event" });

    expect(warn).toHaveBeenCalledWith(
      "Failed to parse WebSocket message as JSON:",
      expect.any(SyntaxError),
    );
    expect(useEventStore.getState().events).toEqual([]);
  });

  it("keeps conversation-scoped effects inactive when no main conversation is selected", () => {
    const consumeMatchingPendingMessage = vi.spyOn(
      useOptimisticUserMessageStore.getState(),
      "consumeMatchingPendingMessage",
    );
    const { queryClient } = renderProvider({ conversationId: undefined });
    const invalidateQueries = vi.spyOn(queryClient, "invalidateQueries");
    useOptimisticUserMessageStore.setState({
      pendingMessages: [
        {
          id: "pending-orphan",
          conversationId: "conv-main",
          text: "orphan",
          content: "orphan",
          status: "sending",
          imageUrls: [],
          fileUrls: [],
          timestamp: "2024-01-01T00:00:00.000Z",
        },
      ],
    });

    dispatchMain(makeMessageEvent("31", "user", ["orphan"]));
    dispatchMain(
      makeActionEvent("32", {
        kind: "ExecuteBashAction",
        command: "echo fallback",
        is_input: false,
        timeout: null,
        reset: false,
      }),
    );
    dispatchMain(
      makeStateEvent("33", "goal", {
        active: true,
        status: "running",
        iteration: 0,
        max_iterations: 2,
        objective: "No owner",
        verdict: null,
      }),
    );
    dispatchMain(
      makeActionEvent(
        "34",
        { kind: "CanvasUIAction", command: "open_tab", tab: "files" },
        "canvas_ui",
      ),
    );

    expect(
      useOptimisticUserMessageStore.getState().pendingMessages,
    ).toHaveLength(1);
    expect(consumeMatchingPendingMessage).not.toHaveBeenCalled();
    expect(useGoalStore.getState().statusByConversation).toEqual({});
    expect(invalidateQueries).toHaveBeenCalledWith(
      { queryKey: ["file_changes", "test-conversation-id"] },
      { cancelRefetch: false },
    );
  });

  it("loads the last planning Plan.md after streamed history completes", async () => {
    vi.spyOn(EventService, "getEventCount").mockResolvedValue(2);
    socketCapture.readConversationFile.mockImplementation(
      (_variables, callbacks) => callbacks.onSuccess("# Loaded plan"),
    );
    renderProvider({
      subConversations: [makeSubConversation()],
      subConversationIds: ["conv-planning"],
    });

    await act(async () => {
      await planningOptions().onOpen?.(new Event("open"));
    });
    expect(screen.getByTestId("loading-history")).toHaveTextContent("true");
    dispatchPlanning(
      makeObservationEvent(
        "35",
        {
          kind: "PlanningFileEditorObservation",
          content: [{ type: "text", text: "history" }],
          is_error: false,
          command: "create",
          path: "/workspace/Plan.md",
          prev_exist: false,
          old_content: null,
          new_content: "history",
        },
        "planning_file_editor",
      ),
    );
    expect(screen.getByTestId("loading-history")).toHaveTextContent("true");
    expect(socketCapture.readConversationFile).not.toHaveBeenCalled();
    dispatchPlanning(makeMessageEvent("36", "assistant", ["planned"]));

    await waitFor(() =>
      expect(screen.getByTestId("loading-history")).toHaveTextContent("false"),
    );
    await waitFor(() =>
      expect(socketCapture.readConversationFile).toHaveBeenCalledWith(
        {
          conversationId: "conv-planning",
          filePath: "/workspace/Plan.md",
        },
        expect.objectContaining({
          onSuccess: expect.any(Function),
          onError: expect.any(Function),
        }),
      ),
    );
    expect(useConversationStore.getState().planContent).toBe("# Loaded plan");
    expect(useEventStore.getState().events).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: "35", isFromPlanningAgent: true }),
        expect.objectContaining({ id: "36", isFromPlanningAgent: true }),
      ]),
    );
  });

  it("finishes planning history when the expected count arrives after its events", async () => {
    vi.spyOn(EventService, "getEventCount").mockResolvedValue(2);
    renderProvider({
      subConversations: [makeSubConversation()],
      subConversationIds: ["conv-planning"],
    });

    dispatchPlanning({ type: "history-envelope" });
    dispatchPlanning({ type: "history-envelope" });
    expect(screen.getByTestId("loading-history")).toHaveTextContent("true");
    expect(useEventStore.getState().events).toEqual([]);

    await act(async () => {
      await planningOptions().onOpen?.(new Event("open"));
    });

    await waitFor(() =>
      expect(screen.getByTestId("loading-history")).toHaveTextContent("false"),
    );
  });

  it("reports a deferred Plan.md read failure after history completes", async () => {
    vi.spyOn(EventService, "getEventCount").mockResolvedValue(1);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    socketCapture.readConversationFile.mockImplementation(
      (_variables, callbacks) =>
        callbacks.onError(new Error("deferred read failed")),
    );
    renderProvider({
      subConversations: [makeSubConversation()],
      subConversationIds: ["conv-planning"],
    });
    await act(async () => {
      await planningOptions().onOpen?.(new Event("open"));
    });

    dispatchPlanning(
      makeObservationEvent(
        "53",
        {
          kind: "PlanningFileEditorObservation",
          content: [],
          is_error: false,
          command: "view",
          path: "/workspace/PLAN.md",
          prev_exist: true,
          old_content: null,
          new_content: null,
        },
        "planning_file_editor",
      ),
    );

    await waitFor(() =>
      expect(warn).toHaveBeenCalledWith(
        "Failed to read conversation file:",
        expect.objectContaining({ message: "deferred read failed" }),
      ),
    );
  });

  it("reads live planning files immediately and reports read failures", async () => {
    vi.spyOn(EventService, "getEventCount").mockResolvedValue(0);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    socketCapture.readConversationFile.mockImplementation(
      (_variables, callbacks) => callbacks.onError(new Error("read failed")),
    );
    renderProvider({
      subConversations: [makeSubConversation()],
      subConversationIds: ["conv-planning"],
    });
    await act(async () => {
      await planningOptions().onOpen?.(new Event("open"));
    });

    dispatchPlanning(
      makeObservationEvent(
        "37",
        {
          kind: "PlanningFileEditorObservation",
          content: [],
          is_error: false,
          command: "view",
          path: "/workspace/PLAN.MD",
          prev_exist: true,
          old_content: null,
          new_content: null,
        },
        "planning_file_editor",
      ),
    );

    expect(socketCapture.readConversationFile).toHaveBeenCalledWith(
      {
        conversationId: "conv-planning",
        filePath: "/workspace/PLAN.MD",
      },
      expect.objectContaining({
        onSuccess: expect.any(Function),
        onError: expect.any(Function),
      }),
    );
    expect(warn).toHaveBeenCalledWith(
      "Failed to read conversation file:",
      expect.objectContaining({ message: "read failed" }),
    );
    warn.mockClear();

    dispatchPlanning(
      makeObservationEvent(
        "37-not-plan",
        {
          kind: "PlanningFileEditorObservation",
          content: [],
          is_error: false,
          command: "view",
          path: "/workspace/notes.md",
          prev_exist: true,
          old_content: null,
          new_content: null,
        },
        "planning_file_editor",
      ),
    );
    dispatchPlanning(
      makeObservationEvent(
        "37-null-path",
        {
          kind: "PlanningFileEditorObservation",
          content: [],
          is_error: false,
          command: "view",
          path: null,
          prev_exist: true,
          old_content: null,
          new_content: null,
        },
        "planning_file_editor",
      ),
    );
    expect(socketCapture.readConversationFile).toHaveBeenCalledOnce();
    expect(warn).not.toHaveBeenCalled();

    socketCapture.readConversationFile.mockImplementation(
      (_variables, callbacks) => callbacks.onSuccess("# Live plan"),
    );
    dispatchPlanning(
      makeObservationEvent(
        "54",
        {
          kind: "PlanningFileEditorObservation",
          content: [],
          is_error: false,
          command: "view",
          path: "/workspace/PLAN.MD",
          prev_exist: true,
          old_content: null,
          new_content: null,
        },
        "planning_file_editor",
      ),
    );
    expect(useConversationStore.getState().planContent).toBe("# Live plan");
  });

  it("falls through planning history when event counting fails", async () => {
    vi.spyOn(EventService, "getEventCount").mockRejectedValue(
      new Error("count unavailable"),
    );
    renderProvider({
      subConversations: [makeSubConversation()],
      subConversationIds: ["conv-planning"],
    });

    await act(async () => {
      await planningOptions().onOpen?.(new Event("open"));
    });

    expect(screen.getByTestId("loading-history")).toHaveTextContent("false");
  });

  it("does not request a planning history count without coordinates", async () => {
    const getEventCount = vi
      .spyOn(EventService, "getEventCount")
      .mockResolvedValue(0);
    renderProvider({
      subConversations: [
        makeSubConversation({ id: "", conversation_url: null }),
      ],
    });

    await act(async () => {
      await planningOptions().onOpen?.(new Event("open"));
    });

    expect(getEventCount).not.toHaveBeenCalled();
  });

  it("does not show a planning connection error before its first open", () => {
    renderProvider({ subConversations: [makeSubConversation()] });

    act(() => planningOptions().onError?.(new Event("error")));

    expect(useErrorMessageStore.getState().errorMessage).toBeNull();
  });

  it("routes planning-agent events to the shared conversation experience", () => {
    const consumeMatchingPendingMessage = vi.spyOn(
      useOptimisticUserMessageStore.getState(),
      "consumeMatchingPendingMessage",
    );
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const { queryClient } = renderProvider({
      subConversations: [makeSubConversation()],
      subConversationIds: ["conv-planning"],
    });
    const invalidateQueries = vi.spyOn(queryClient, "invalidateQueries");

    useErrorMessageStore.getState().setErrorMessage("temporary", "connection");
    dispatchPlanning(makeMessageEvent("37-normal", "assistant", ["ready"]));
    expect(useErrorMessageStore.getState().errorMessage).toBeNull();
    expect(consumeMatchingPendingMessage).not.toHaveBeenCalled();
    expect(socketCapture.trackError).not.toHaveBeenCalled();
    expect(warn).not.toHaveBeenCalled();

    dispatchPlanning({
      ...baseEvent("38"),
      kind: "ServerErrorEvent",
      code: "PlannerUnavailable",
      detail: "Planner failed",
    });
    expect(useErrorMessageStore.getState()).toMatchObject({
      errorMessage: "Planner failed",
      errorType: "conversation",
      errorCode: "PlannerUnavailable",
    });
    expect(socketCapture.trackError).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "planning_conversation",
        metadata: expect.objectContaining({
          eventId: "38",
          errorCode: "PlannerUnavailable",
        }),
      }),
    );

    dispatchPlanning({
      ...baseEvent("39", "agent"),
      tool_name: "planner_llm",
      tool_call_id: "call-39",
      error: "Planner model failed",
    });
    expect(socketCapture.trackError).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "planning_agent",
        metadata: expect.objectContaining({
          eventId: "39",
          toolName: "planner_llm",
          toolCallId: "call-39",
        }),
      }),
    );
    expect(socketCapture.trackError).toHaveBeenCalledTimes(2);

    useOptimisticUserMessageStore.setState({
      pendingMessages: [
        {
          id: "pending-plan",
          conversationId: "conv-main",
          text: "plan this",
          content: "plan this",
          status: "sending",
          imageUrls: [],
          fileUrls: [],
          timestamp: "2024-01-01T00:00:00.000Z",
        },
      ],
    });
    setConversationState("conv-main", { draftMessage: "plan this" });
    dispatchPlanning(makeMessageEvent("40", "user", ["plan this"]));
    expect(useOptimisticUserMessageStore.getState().pendingMessages).toEqual(
      [],
    );
    expect(consumeMatchingPendingMessage).toHaveBeenCalledOnce();
    expect(consumeMatchingPendingMessage).toHaveBeenCalledWith(
      "conv-main",
      "plan this",
    );
    expect(getConversationState("conv-main").draftMessage).toBeNull();

    dispatchPlanning(
      makeActionEvent("41", {
        kind: "ExecuteBashAction",
        command: "cat PLAN.md",
        is_input: false,
        timeout: null,
        reset: false,
      }),
    );
    expect(useCommandStore.getState().commands).toContainEqual({
      type: "input",
      content: "cat PLAN.md",
    });
    expect(invalidateQueries).toHaveBeenCalledWith(
      { queryKey: ["file_changes", "conv-planning"] },
      { cancelRefetch: false },
    );

    dispatchPlanning(
      makeStateEvent("42", "full_state", {
        execution_status: "running",
      }),
    );
    // The planning socket scopes execution status to the planning agent's own
    // conversation id, never the main conversation.
    expect(
      useConversationStateStore.getState().executionStatusByConversation[
        "conv-planning"
      ],
    ).toBe("running");
    dispatchPlanning(makeStateEvent("43", "execution_status", "finished"));
    expect(
      useConversationStateStore.getState().executionStatusByConversation[
        "conv-planning"
      ],
    ).toBe("finished");
    dispatchPlanning(
      makeStateEvent("44", "stats", {
        usage_to_metrics: {
          agent: {
            accumulated_cost: 4,
            max_budget_per_task: 10,
            accumulated_token_usage: null,
          },
        },
      }),
    );
    expect(useMetricsStore.getState().cost).toBe(4);
    expect(
      useConversationStateStore.getState().executionStatusByConversation[
        "conv-planning"
      ],
    ).toBe("finished");

    const planningGoal = {
      active: true,
      status: "running",
      iteration: 1,
      max_iterations: 4,
      objective: "Plan",
      verdict: null,
    };
    dispatchPlanning(makeStateEvent("45", "goal", planningGoal));
    expect(useGoalStore.getState().statusByConversation["conv-main"]).toEqual(
      planningGoal,
    );
    expect(
      useConversationStateStore.getState().executionStatusByConversation[
        "conv-planning"
      ],
    ).toBe("finished");

    dispatchPlanning(
      makeObservationEvent("46", {
        kind: "ExecuteBashObservation",
        content: [
          { type: "text", text: "plan output" },
          { type: "image", image_urls: [] },
          { type: "text", text: "second line" },
        ],
      }),
    );
    expect(useCommandStore.getState().commands).toContainEqual({
      type: "output",
      content: "plan output\nsecond line",
    });

    dispatchPlanning(
      makeObservationEvent(
        "47",
        {
          kind: "PlanningFileEditorObservation",
          content: [],
          is_error: false,
          command: "view",
          path: "/workspace/notes.md",
          prev_exist: true,
          old_content: null,
          new_content: null,
        },
        "planning_file_editor",
      ),
    );
    expect(socketCapture.readConversationFile).not.toHaveBeenCalled();
    expect(warn).not.toHaveBeenCalled();
  });

  it("keeps planning fallbacks safe without conversation identities", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const { queryClient } = renderProvider({
      conversationId: undefined,
      subConversations: undefined,
      subConversationIds: undefined,
    });
    const invalidateQueries = vi.spyOn(queryClient, "invalidateQueries");

    await act(async () => {
      await planningOptions().onOpen?.(new Event("open"));
    });
    dispatchPlanning(makeMessageEvent("48", "user", ["unscoped"]));
    dispatchPlanning(
      makeActionEvent("49", {
        kind: "ExecuteBashAction",
        command: "echo fallback",
        is_input: false,
        timeout: null,
        reset: false,
      }),
    );
    dispatchPlanning(
      makeStateEvent("50", "goal", {
        active: false,
        status: "interrupted",
        iteration: 1,
        max_iterations: 2,
        objective: "No owner",
        verdict: null,
      }),
    );
    dispatchPlanning(
      makeObservationEvent(
        "51",
        {
          kind: "PlanningFileEditorObservation",
          content: [],
          is_error: false,
          command: "view",
          path: null,
          prev_exist: true,
          old_content: null,
          new_content: null,
        },
        "planning_file_editor",
      ),
    );
    dispatchPlanning(
      makeObservationEvent(
        "52",
        {
          kind: "PlanningFileEditorObservation",
          content: [],
          is_error: false,
          command: "view",
          path: "/workspace/PLAN.md",
          prev_exist: true,
          old_content: null,
          new_content: null,
        },
        "planning_file_editor",
      ),
    );
    expect(warn).not.toHaveBeenCalled();
    act(() =>
      planningOptions().onMessage?.({ data: "not-json" } as MessageEvent),
    );

    expect(invalidateQueries).toHaveBeenCalledWith(
      { queryKey: ["file_changes", "test-conversation-id"] },
      { cancelRefetch: false },
    );
    expect(useGoalStore.getState().statusByConversation).toEqual({});
    expect(socketCapture.readConversationFile).not.toHaveBeenCalled();
    expect(warn).toHaveBeenCalledWith(
      "Failed to parse WebSocket message as JSON:",
      expect.any(SyntaxError),
    );
    expect(warn).toHaveBeenCalledTimes(1);
  });
});
