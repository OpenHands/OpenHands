import { renderHook, act } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { useHandleWSEvents } from "#/hooks/use-handle-ws-events";
import { useEventStore } from "#/stores/use-event-store";
import { useSendMessage } from "#/hooks/use-send-message";

vi.mock("#/hooks/use-send-message", () => ({
  useSendMessage: vi.fn(),
}));
vi.mock("#/services/agent-state-service", () => ({
  generateAgentStateChangeEvent: vi.fn(() => ({
    type: "agent_state_change",
  })),
}));
vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
}));

import { useSendMessage as useSendMessageImpl } from "#/hooks/use-send-message";

describe("useHandleWSEvents", () => {
  const mockSend = vi.fn();
  const mockGenerateEvent = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockSend.mockReset();
    useEventStore.setState({ events: [] });
    (useSendMessageImpl as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      send: mockSend,
    });
  });

  afterEach(() => {
    useEventStore.setState({ events: [] });
  });

  it("should call send with PAUSED state change when Agent reached maximum error arrives", () => {
    // Two distinct setState calls ensure events.length changes each time,
    // so the effect fires after the error event is added.
    act(() => {
      useEventStore.setState({ events: [{ type: "info" as never }] });
    });
    act(() => {
      useEventStore.setState({
        events: [
          { type: "info" as never },
          { type: "error", message: "Agent reached maximum" },
        ],
      });
    });
    expect(mockSend).toHaveBeenCalledWith(
      expect.objectContaining({ type: "agent_state_change" }),
    );
  });

  it("should use latest send after reconnect (no stale closure)", () => {
    // Simulate two different send functions from two different re-renders
    const sendV1 = vi.fn();
    const sendV2 = vi.fn();
    const setMock = vi.fn((fn) => {
      if (typeof fn === "function") {
        fn({ send: sendV2 });
      }
    });
    (useSendMessageImpl as unknown as ReturnType<typeof vi.fn>).mockImplementation(
      setMock,
    );

    const { rerender } = renderHook(() => useHandleWSEvents());
    act(() => {
      useEventStore.setState({
        events: [{ type: "info" as never }],
      });
    });

    // Simulate reconnect causing a new useSendMessage call
    rerender();

    act(() => {
      useEventStore.setState({
        events: [
          { type: "info" as never },
          { type: "error", message: "Agent reached maximum" },
        ],
      });
    });
    // Should call the latest send, not the stale one
    expect(sendV2).toHaveBeenCalled();
    expect(sendV1).not.toHaveBeenCalled();
  });
});
