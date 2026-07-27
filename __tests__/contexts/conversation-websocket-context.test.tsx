import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, waitFor, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createUserMessageEvent } from "test-utils";
import { ConversationWebSocketProvider } from "#/contexts/conversation-websocket-context";
import { useEventStore } from "#/stores/use-event-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { useBrowserStore } from "#/stores/browser-store";
import { useCommandStore } from "#/stores/command-store";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { useUserConversation } from "#/hooks/query/use-user-conversation";
import EventService from "#/api/event-service/event-service.api";
import {
  getStoredConversationMetadata,
  setStoredConversationMetadata,
} from "#/api/conversation-metadata-store";
import type { MessageEvent } from "#/types/agent-server/core";

// Captures the main socket's `onMessage` (`handleMainMessage`) so tests can
// drive the live message path without a real WebSocket. Only the main socket
// gets a non-empty url (planning stays ""), so url presence discriminates it.
const wsCapture = vi.hoisted(() => ({
  mainOnMessage: null as null | ((event: { data: string }) => void),
}));

// Keep the units under test real (the provider, `useConversationHistory`, the
// event store). Only the network is stubbed: the WebSocket transport and the
// REST service the history query depends on.
vi.mock("#/hooks/use-websocket", () => ({
  useWebSocket: vi.fn(
    (
      url: string,
      options?: { onMessage?: (event: { data: string }) => void },
    ) => {
      if (url && options?.onMessage) {
        wsCapture.mainOnMessage = options.onMessage;
      }
      return { socket: null, reconnect: vi.fn() };
    },
  ),
}));
vi.mock("#/hooks/query/use-user-conversation", () => ({
  useUserConversation: vi.fn(),
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
  activated_microagents: [],
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

    const makeConversationError = (id: string, detail: string) => ({
      id,
      timestamp: new Date().toISOString(),
      source: "environment",
      kind: "ConversationErrorEvent",
      detail,
      code: "SomeError",
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
