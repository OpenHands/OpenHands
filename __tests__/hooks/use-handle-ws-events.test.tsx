import * as ReactModule from "react";
import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useHandleWSEvents } from "#/hooks/use-handle-ws-events";

const { eventStoreState, pauseMutateMock, displayErrorToastMock } = vi.hoisted(
  () => ({
    eventStoreState: { events: [] as object[] },
    pauseMutateMock: vi.fn(),
    displayErrorToastMock: vi.fn(),
  }),
);

vi.mock("#/stores/use-event-store", () => ({
  useEventStore: (selector: (state: { events: object[] }) => unknown) =>
    selector(eventStoreState),
}));

vi.mock("#/hooks/mutation/use-pause-conversation", () => ({
  usePauseConversation: () => ({ mutate: pauseMutateMock }),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "conversation-1" }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: (...args: unknown[]) => displayErrorToastMock(...args),
}));

function handle(event?: object) {
  eventStoreState.events = event ? [event] : [];
  return renderHook(() => useHandleWSEvents());
}

beforeEach(() => {
  vi.clearAllMocks();
  eventStoreState.events = [];
});

describe("WebSocket error event handling", () => {
  it("does nothing before any events arrive", () => {
    handle();

    expect(displayErrorToastMock).not.toHaveBeenCalled();
    expect(pauseMutateMock).not.toHaveBeenCalled();
  });

  it("keeps V1 agent errors inline without showing a duplicate toast", () => {
    handle({
      id: "agent-error-1",
      timestamp: "2026-07-12T00:00:00.000Z",
      source: "agent",
      tool_name: "terminal",
      tool_call_id: "tool-call-1",
      error: "command failed",
    });

    expect(displayErrorToastMock).not.toHaveBeenCalled();
    expect(pauseMutateMock).not.toHaveBeenCalled();
  });

  it("keeps hook execution events inline without showing a toast", () => {
    // Shape of a hook event as the app server serves it from persisted
    // history: `error` is present but null.
    handle({
      id: "hook-1",
      timestamp: "2026-07-12T00:00:00.000Z",
      source: "hook",
      kind: "HookExecutionEvent",
      hook_event_type: "Stop",
      hook_command: "python finish_tool_hook.py",
      success: true,
      blocked: false,
      exit_code: 0,
      reason: null,
      tool_name: null,
      action_id: null,
      message_id: null,
      stdout: "",
      stderr: "",
      error: null,
      additional_context: null,
      hook_input: { reason: "agent_finished" },
    });

    expect(displayErrorToastMock).not.toHaveBeenCalled();
    expect(pauseMutateMock).not.toHaveBeenCalled();
  });

  it("shows the session-expired message for 401 server errors", () => {
    handle({
      error: "raw unauthorized error",
      error_code: 401,
      message: "Unauthorized",
    });

    expect(displayErrorToastMock).toHaveBeenCalledWith("Session expired.");
    expect(displayErrorToastMock).toHaveBeenCalledOnce();
  });

  it("shows a string server error directly", () => {
    handle({ error: "Runtime disconnected", message: "fallback" });

    expect(displayErrorToastMock).toHaveBeenCalledWith("Runtime disconnected");
  });

  it("uses the server message when the error flag is not a string", () => {
    handle({ error: true, message: "Sandbox failed to start" });

    expect(displayErrorToastMock).toHaveBeenCalledWith(
      "Sandbox failed to start",
    );
  });

  it("reacts when the event list changes after the hook mounts", () => {
    const { rerender } = handle();
    expect(displayErrorToastMock).not.toHaveBeenCalled();

    eventStoreState.events = [
      { error: "Late runtime failure", message: "fallback" },
    ];
    rerender();

    expect(displayErrorToastMock).toHaveBeenCalledWith("Late runtime failure");
  });

  const priorEvent = {
    id: "message-1",
    timestamp: "2026-07-12T00:00:00.000Z",
    source: "user",
    kind: "MessageEvent",
  };

  it("pauses the conversation when a max-iterations error arrives live", () => {
    const { rerender } = handle(priorEvent);
    expect(pauseMutateMock).not.toHaveBeenCalled();

    eventStoreState.events = [
      priorEvent,
      {
        id: "conversation-error-1",
        timestamp: "2026-07-12T00:00:00.000Z",
        source: "environment",
        kind: "ConversationErrorEvent",
        code: "MaxIterationsReached",
        detail: "Agent reached maximum iterations limit (500).",
      },
    ];
    rerender();

    expect(pauseMutateMock).toHaveBeenCalledWith({
      conversationId: "conversation-1",
    });
    expect(pauseMutateMock).toHaveBeenCalledOnce();
  });

  it("does not pause for a max-iterations error already in the store on mount", () => {
    // A cap hit from a previous session is replayed from history when the
    // conversation is reopened; it is not a live arrival to react to.
    handle({
      id: "conversation-error-1",
      timestamp: "2026-07-12T00:00:00.000Z",
      source: "environment",
      kind: "ConversationErrorEvent",
      code: "MaxIterationsReached",
      detail: "Agent reached maximum iterations limit (500).",
    });

    expect(pauseMutateMock).not.toHaveBeenCalled();
  });

  it("does not pause for a conversation error with another code", () => {
    const { rerender } = handle(priorEvent);

    eventStoreState.events = [
      priorEvent,
      {
        id: "conversation-error-2",
        timestamp: "2026-07-12T00:00:00.000Z",
        source: "environment",
        kind: "ConversationErrorEvent",
        code: "LLMAuthenticationError",
        detail: "Your LLM API key appears to be invalid or has expired.",
      },
    ];
    rerender();

    expect(pauseMutateMock).not.toHaveBeenCalled();
  });

  it("does not pause for a planning agent's max-iterations error", () => {
    const { rerender } = handle(priorEvent);

    eventStoreState.events = [
      priorEvent,
      {
        id: "conversation-error-3",
        timestamp: "2026-07-12T00:00:00.000Z",
        source: "environment",
        kind: "ConversationErrorEvent",
        code: "MaxIterationsReached",
        detail: "Agent reached maximum iterations limit (500).",
        isFromPlanningAgent: true,
      },
    ];
    rerender();

    expect(pauseMutateMock).not.toHaveBeenCalled();
  });

  it("ignores the legacy V0 typed error shape", () => {
    const { rerender } = handle(priorEvent);

    eventStoreState.events = [
      priorEvent,
      {
        type: "error",
        message: "Agent reached maximum iteration limit",
      },
    ];
    rerender();

    expect(pauseMutateMock).not.toHaveBeenCalled();
    expect(displayErrorToastMock).not.toHaveBeenCalled();
  });

  it("ignores ordinary agent-server events", () => {
    handle({
      id: "message-1",
      timestamp: "2026-07-12T00:00:00.000Z",
      source: "user",
      llm_message: { role: "user", content: [{ type: "text", text: "hi" }] },
    });

    expect(pauseMutateMock).not.toHaveBeenCalled();
    expect(displayErrorToastMock).not.toHaveBeenCalled();
  });

  it("initializes fresh server and conversation error classifiers", async () => {
    vi.doMock("react", () => ReactModule);
    vi.resetModules();

    try {
      const { useHandleWSEvents: freshHook } =
        await import("#/hooks/use-handle-ws-events");

      eventStoreState.events = [
        { error: "Fresh server failure", message: "fallback" },
      ];
      const serverRender = renderHook(() => freshHook());
      expect(displayErrorToastMock).toHaveBeenCalledWith(
        "Fresh server failure",
      );
      serverRender.unmount();

      vi.clearAllMocks();
      eventStoreState.events = [priorEvent];
      const typedRender = renderHook(() => freshHook());
      eventStoreState.events = [
        priorEvent,
        {
          id: "conversation-error-fresh",
          timestamp: "2026-07-12T00:00:00.000Z",
          source: "environment",
          kind: "ConversationErrorEvent",
          code: "MaxIterationsReached",
          detail: "Agent reached maximum iterations limit (500).",
        },
      ];
      typedRender.rerender();
      expect(pauseMutateMock).toHaveBeenCalledWith({
        conversationId: "conversation-1",
      });
      typedRender.unmount();
    } finally {
      vi.doUnmock("react");
      vi.resetModules();
    }
  });
});
