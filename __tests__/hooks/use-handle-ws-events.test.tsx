import * as ReactModule from "react";
import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useHandleWSEvents } from "#/hooks/use-handle-ws-events";

const { eventStoreState, sendMock, displayErrorToastMock } = vi.hoisted(() => ({
  eventStoreState: { events: [] as object[] },
  sendMock: vi.fn(),
  displayErrorToastMock: vi.fn(),
}));

vi.mock("#/stores/use-event-store", () => ({
  useEventStore: (selector: (state: { events: object[] }) => unknown) =>
    selector(eventStoreState),
}));

vi.mock("#/hooks/use-send-message", () => ({
  useSendMessage: () => ({ send: sendMock }),
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
    expect(sendMock).not.toHaveBeenCalled();
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
    expect(sendMock).not.toHaveBeenCalled();
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

  it("pauses the agent after a maximum-iteration typed error", () => {
    handle({
      id: "typed-error-1",
      timestamp: "2026-07-12T00:00:00.000Z",
      source: "agent",
      type: "error",
      message: "Agent reached maximum iteration limit",
    });

    expect(sendMock).toHaveBeenCalledWith({
      action: "change_agent_state",
      args: { agent_state: "paused" },
    });
    expect(displayErrorToastMock).not.toHaveBeenCalled();
  });

  it("does not pause for another typed error", () => {
    handle({ type: "error", message: "Connection interrupted" });

    expect(sendMock).not.toHaveBeenCalled();
    expect(displayErrorToastMock).not.toHaveBeenCalled();
  });

  it("handles a typed error without a message", () => {
    handle({ type: "error" });

    expect(sendMock).not.toHaveBeenCalled();
    expect(displayErrorToastMock).not.toHaveBeenCalled();
  });

  it("ignores non-error typed events", () => {
    handle({ type: "message", message: "Agent reached maximum" });

    expect(sendMock).not.toHaveBeenCalled();
    expect(displayErrorToastMock).not.toHaveBeenCalled();
  });

  it("ignores ordinary agent-server events", () => {
    handle({
      id: "message-1",
      timestamp: "2026-07-12T00:00:00.000Z",
      source: "user",
      llm_message: { role: "user", content: [{ type: "text", text: "hi" }] },
    });

    expect(sendMock).not.toHaveBeenCalled();
    expect(displayErrorToastMock).not.toHaveBeenCalled();
  });

  it("initializes fresh server and typed error classifiers", async () => {
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
      eventStoreState.events = [
        {
          type: "error",
          message: "Agent reached maximum iteration limit",
        },
      ];
      const typedRender = renderHook(() => freshHook());
      expect(sendMock).toHaveBeenCalledWith({
        action: "change_agent_state",
        args: { agent_state: "paused" },
      });
      typedRender.unmount();
    } finally {
      vi.doUnmock("react");
      vi.resetModules();
    }
  });
});
