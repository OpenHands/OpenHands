import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import { ConversationConfirmationButtons } from "#/components/shared/buttons/conversation-confirmation-buttons";
import { AgentState } from "#/types/agent-state";
import { SecurityRisk } from "#/types/agent-server/core/base/common";

const {
  eventMessageState,
  eventState,
  activeConversationState,
  agentState,
  addSubmittedEventIdMock,
  respondToConfirmationMock,
} = vi.hoisted(() => ({
  eventMessageState: { submittedEventIds: [] as Array<string | number> },
  eventState: { events: [] as object[] },
  activeConversationState: {
    data: undefined as Record<string, unknown> | undefined,
  },
  agentState: { curAgentState: "running" },
  addSubmittedEventIdMock: vi.fn(),
  respondToConfirmationMock: vi.fn(),
}));

vi.mock("react-i18next", () => ({
  useTranslation: (namespace: string) => ({
    t: (key: string) =>
      namespace === "openhands" ? key : `wrong-namespace:${key}`,
  }),
}));

vi.mock("#/components/shared/action-tooltip", () => ({
  ActionTooltip: ({
    type,
    onClick,
  }: {
    type: "confirm" | "reject";
    onClick: () => void;
  }) => (
    <button data-testid={`action-${type}-button`} onClick={onClick}>
      {type}
    </button>
  ),
}));

vi.mock("#/components/shared/risk-alert", () => ({
  RiskAlert: ({ content, title }: { content: ReactNode; title: string }) => (
    <div data-testid="risk-alert">
      {title}: {content}
    </div>
  ),
}));

vi.mock("#/stores/event-message-store", () => ({
  useEventMessageStore: (
    selector: (state: {
      submittedEventIds: Array<string | number>;
      addSubmittedEventId: typeof addSubmittedEventIdMock;
    }) => unknown,
  ) =>
    selector({
      ...eventMessageState,
      addSubmittedEventId: addSubmittedEventIdMock,
    }),
}));

vi.mock("#/stores/use-event-store", () => ({
  useEventStore: (selector: (state: { events: object[] }) => unknown) =>
    selector(eventState),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => activeConversationState,
}));

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => agentState,
}));

vi.mock("#/hooks/mutation/use-respond-to-confirmation", () => ({
  useRespondToConfirmation: () => ({ mutate: respondToConfirmationMock }),
}));

const actionEvent = (
  id: string,
  securityRisk: SecurityRisk = SecurityRisk.LOW,
) => ({
  id,
  timestamp: "2026-07-12T00:00:00.000Z",
  source: "agent",
  tool_name: "terminal",
  tool_call_id: `tool-${id}`,
  security_risk: securityRisk,
  action: { kind: "ExecuteBashAction", command: "pwd" },
});

const userEvent = (id: string) => ({
  id,
  timestamp: "2026-07-12T00:00:01.000Z",
  source: "user",
});

const setupState = () => {
  vi.clearAllMocks();
  eventMessageState.submittedEventIds = [];
  eventState.events = [];
  activeConversationState.data = {
    id: "conv-1",
    conversation_url: "https://runtime.example.com",
    session_api_key: "session-key",
  };
  agentState.curAgentState = AgentState.AWAITING_USER_CONFIRMATION;
};

describe("conversation confirmation controls", () => {
  it("stays hidden when the agent is not awaiting confirmation", () => {
    setupState();
    eventState.events = [actionEvent("action-1")];
    agentState.curAgentState = AgentState.RUNNING;
    const addEventListener = vi.spyOn(document, "addEventListener");

    render(<ConversationConfirmationButtons />);
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", metaKey: true }),
    );

    expect(
      screen.queryByTestId("action-confirm-button"),
    ).not.toBeInTheDocument();
    expect(
      addEventListener.mock.calls.filter(
        ([eventName]) => eventName === "keydown",
      ),
    ).toHaveLength(0);
    expect(respondToConfirmationMock).not.toHaveBeenCalled();
  });

  it("stays hidden when no agent event is available", () => {
    setupState();
    eventState.events = [userEvent("user-1")];

    render(<ConversationConfirmationButtons />);

    expect(
      screen.queryByTestId("action-confirm-button"),
    ).not.toBeInTheDocument();
  });

  it("stays hidden after the awaiting event has already been submitted", () => {
    setupState();
    eventState.events = [actionEvent("action-1")];
    eventMessageState.submittedEventIds = ["action-1"];

    render(<ConversationConfirmationButtons />);

    expect(
      screen.queryByTestId("action-confirm-button"),
    ).not.toBeInTheDocument();
  });

  it("uses the latest agent event and submits rejection", () => {
    setupState();
    const olderAction = actionEvent("older-action");
    const latestAction = actionEvent("latest-action");
    eventState.events = [olderAction, latestAction];

    render(<ConversationConfirmationButtons />);
    expect(eventState.events).toEqual([olderAction, latestAction]);
    fireEvent.click(screen.getByTestId("action-reject-button"));

    expect(screen.queryByTestId("risk-alert")).not.toBeInTheDocument();
    expect(addSubmittedEventIdMock).toHaveBeenCalledWith("latest-action");
    expect(respondToConfirmationMock).toHaveBeenCalledWith({
      conversationId: "conv-1",
      conversationUrl: "https://runtime.example.com",
      sessionApiKey: "session-key",
      accept: false,
    });
  });

  it("shows high-risk context and submits acceptance with a URL fallback", () => {
    setupState();
    eventState.events = [actionEvent("risky-action", SecurityRisk.HIGH)];
    activeConversationState.data = {
      id: "conv-1",
      conversation_url: null,
      session_api_key: null,
    };

    render(<ConversationConfirmationButtons />);

    expect(screen.getByTestId("risk-alert")).toHaveTextContent(
      "COMMON$HIGH_RISK: CHAT_INTERFACE$HIGH_RISK_WARNING",
    );
    fireEvent.click(screen.getByTestId("action-confirm-button"));
    expect(addSubmittedEventIdMock).toHaveBeenCalledWith("risky-action");
    expect(respondToConfirmationMock).toHaveBeenCalledWith({
      conversationId: "conv-1",
      conversationUrl: "",
      sessionApiKey: null,
      accept: true,
    });
  });

  it("renders unknown risk for a non-action agent event without an alert", () => {
    setupState();
    eventState.events = [
      {
        id: "agent-message",
        timestamp: "2026-07-12T00:00:00.000Z",
        source: "agent",
      },
    ];

    render(<ConversationConfirmationButtons />);

    expect(screen.getByTestId("action-confirm-button")).toBeInTheDocument();
    expect(screen.queryByTestId("risk-alert")).not.toBeInTheDocument();
  });

  it("does not submit without active conversation metadata", () => {
    setupState();
    eventState.events = [actionEvent("action-1")];
    activeConversationState.data = undefined;

    render(<ConversationConfirmationButtons />);
    fireEvent.click(screen.getByTestId("action-confirm-button"));

    expect(addSubmittedEventIdMock).not.toHaveBeenCalled();
    expect(respondToConfirmationMock).not.toHaveBeenCalled();
  });

  it("uses the latest event and conversation metadata after rerender", () => {
    setupState();
    eventState.events = [actionEvent("old-action")];
    const { rerender } = render(<ConversationConfirmationButtons />);

    eventState.events = [actionEvent("latest-action")];
    activeConversationState.data = {
      id: "conv-2",
      conversation_url: "https://latest-runtime.example.com",
      session_api_key: "latest-session-key",
    };
    rerender(<ConversationConfirmationButtons />);
    document.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Enter",
        metaKey: true,
        cancelable: true,
      }),
    );

    expect(addSubmittedEventIdMock).toHaveBeenCalledWith("latest-action");
    expect(respondToConfirmationMock).toHaveBeenCalledWith({
      conversationId: "conv-2",
      conversationUrl: "https://latest-runtime.example.com",
      sessionApiKey: "latest-session-key",
      accept: true,
    });
  });

  it("supports cancel and continue keyboard shortcuts and ignores near misses", () => {
    setupState();
    eventState.events = [actionEvent("action-1")];
    const { unmount } = render(<ConversationConfirmationButtons />);

    const wrongCancel = new KeyboardEvent("keydown", {
      key: "Backspace",
      shiftKey: true,
      metaKey: false,
      cancelable: true,
    });
    document.dispatchEvent(wrongCancel);
    const otherKey = new KeyboardEvent("keydown", {
      key: "Escape",
      shiftKey: true,
      metaKey: true,
      cancelable: true,
    });
    document.dispatchEvent(otherKey);
    const cancel = new KeyboardEvent("keydown", {
      key: "Backspace",
      shiftKey: true,
      metaKey: true,
      cancelable: true,
    });
    document.dispatchEvent(cancel);
    const continueEvent = new KeyboardEvent("keydown", {
      key: "Enter",
      metaKey: true,
      cancelable: true,
    });
    document.dispatchEvent(continueEvent);

    expect(wrongCancel.defaultPrevented).toBe(false);
    expect(otherKey.defaultPrevented).toBe(false);
    expect(cancel.defaultPrevented).toBe(true);
    expect(continueEvent.defaultPrevented).toBe(true);
    expect(respondToConfirmationMock).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({ accept: false }),
    );
    expect(respondToConfirmationMock).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ accept: true }),
    );

    unmount();
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", metaKey: true }),
    );
    expect(respondToConfirmationMock).toHaveBeenCalledTimes(2);
  });
});
