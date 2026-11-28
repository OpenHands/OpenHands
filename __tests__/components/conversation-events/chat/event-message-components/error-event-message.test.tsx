import { render, screen, fireEvent } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ErrorEventMessage } from "#/components/conversation-events/chat/event-message-components/error-event-message";
import { AgentState } from "#/types/agent-state";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { useEventStore } from "#/stores/use-event-store";
import type { AgentErrorEvent } from "#/types/agent-server/core";

const mockMutate = vi.fn();
vi.mock("#/hooks/mutation/use-resume-conversation", () => ({
  useResumeConversation: () => ({
    mutate: mockMutate,
    isPending: false,
  }),
}));

const mockUseOptionalConversationId = vi.fn();
vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => mockUseOptionalConversationId(),
}));

const mockUseAgentState = vi.fn();
vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => mockUseAgentState(),
}));

const mockUseIsArchivedConversation = vi.fn();
vi.mock("#/hooks/use-is-archived-conversation", () => ({
  useIsArchivedConversation: () => mockUseIsArchivedConversation(),
}));

const createMockEvent = (
  overrides?: Partial<AgentErrorEvent>,
): AgentErrorEvent =>
  ({
    id: "error-1",
    timestamp: "2026-11-20T10:00:00Z",
    source: "agent",
    tool_name: "execute_bash",
    error: "Agent encountered a fatal error",
    tool_call_id: "tool-1",
    ...overrides,
  }) as unknown as AgentErrorEvent;

describe("ErrorEventMessage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseOptionalConversationId.mockReturnValue({
      conversationId: "conv-123",
    });
    mockUseAgentState.mockReturnValue({
      curAgentState: AgentState.ERROR,
    });
    mockUseIsArchivedConversation.mockReturnValue(false);
    useErrorMessageStore.setState({ errorMessage: "Active error" });
    useEventStore.setState({
      uiEvents: [createMockEvent()],
    });
  });

  it("renders retry button when error is latest, retryable, and agent is in ERROR state", () => {
    render(<ErrorEventMessage event={createMockEvent()} />);

    expect(screen.getByTestId("error-message-retry")).toBeInTheDocument();
  });

  it("does not render retry button when event classification is non-retryable", () => {
    const event = createMockEvent({
      classification: {
        kind: "auth",
        retryable: false,
        user_action: "settings",
      },
    });
    useEventStore.setState({ uiEvents: [event] });

    render(<ErrorEventMessage event={event} />);

    expect(screen.queryByTestId("error-message-retry")).not.toBeInTheDocument();
  });

  it("does not render retry button when conversation is archived", () => {
    mockUseIsArchivedConversation.mockReturnValue(true);

    render(<ErrorEventMessage event={createMockEvent()} />);

    expect(screen.queryByTestId("error-message-retry")).not.toBeInTheDocument();
  });

  it("does not render retry button when event is not the latest error", () => {
    const olderEvent = createMockEvent({ id: "error-older" });
    const newerEvent = createMockEvent({ id: "error-newer" });
    useEventStore.setState({ uiEvents: [olderEvent, newerEvent] });

    render(<ErrorEventMessage event={olderEvent} />);

    expect(screen.queryByTestId("error-message-retry")).not.toBeInTheDocument();
  });

  it("calls resume mutation and clears error message only on success", () => {
    render(<ErrorEventMessage event={createMockEvent()} />);

    const retryBtn = screen.getByTestId("error-message-retry");
    fireEvent.click(retryBtn);

    expect(mockMutate).toHaveBeenCalledWith(
      { conversationId: "conv-123" },
      expect.objectContaining({ onSuccess: expect.any(Function) }),
    );

    // Error message remains until onSuccess fires
    expect(useErrorMessageStore.getState().errorMessage).toBe("Active error");

    // Trigger onSuccess callback
    const options = mockMutate.mock.calls[0][1];
    options.onSuccess();
    expect(useErrorMessageStore.getState().errorMessage).toBeNull();
  });
});
