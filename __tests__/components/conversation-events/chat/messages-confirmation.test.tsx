import { describe, expect, it, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import { Messages } from "#/components/conversation-events/chat/messages";
import { useEventStore } from "#/stores/use-event-store";
import { AgentState } from "#/types/agent-state";
import { ActionEvent, SecurityRisk } from "#/types/agent-server/core";
import { ExecuteBashAction } from "#/types/agent-server/core/base/action";
import { renderWithProviders } from "test-utils";

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({ data: {} }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      id: "test-conversation-id",
      conversation_url: "",
      session_api_key: null,
    },
  }),
}));

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => ({
    curAgentState: AgentState.AWAITING_USER_CONFIRMATION,
  }),
}));

vi.mock("#/hooks/mutation/use-respond-to-confirmation", () => ({
  useRespondToConfirmation: () => ({ mutate: vi.fn() }),
}));

const createBashActionEvent = (
  id: string,
  command: string,
): ActionEvent<ExecuteBashAction> => ({
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
  tool_call_id: `call_bash_${id}`,
  tool_call: {
    id: `call_bash_${id}`,
    type: "function",
    function: {
      name: "execute_bash",
      arguments: JSON.stringify({ command }),
    },
  },
  llm_response_id: `response_${id}`,
  security_risk: SecurityRisk.HIGH,
});

describe("Messages confirmation prompt", () => {
  beforeEach(() => {
    useEventStore.setState({
      events: [],
      eventIds: new Set(),
      uiEvents: [],
      loadedConversationId: null,
    });
  });

  it("keeps confirmation visible when the latest action is inside a collapsed event group", () => {
    const events = [
      createBashActionEvent("action-1", "echo first"),
      createBashActionEvent("action-2", "echo second"),
    ];

    useEventStore.setState({
      events,
      eventIds: new Set(events.map((event) => event.id)),
      uiEvents: events,
      loadedConversationId: "test-conversation-id",
    });

    renderWithProviders(<Messages messages={events} allEvents={events} />);

    expect(screen.getByTestId("event-group")).toBeInTheDocument();
    expect(screen.getByTestId("event-group-toggle")).toHaveAttribute(
      "aria-expanded",
      "false",
    );
    expect(
      screen.getByText("CHAT_INTERFACE$USER_ASK_CONFIRMATION"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("action-reject-button")).toBeInTheDocument();
    expect(screen.getByTestId("action-confirm-button")).toBeInTheDocument();
  });
});
