import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import { EventMessage } from "#/components/v1/chat/event-message";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";
import { ActionEvent, SecurityRisk } from "#/types/v1/core";
import { ThinkAction, ExecuteBashAction } from "#/types/v1/core/base/action";
import { renderWithProviders } from "test-utils";

// Mock useConfig
vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({
    data: { APP_MODE: "saas" },
  }),
}));

// Mock useAgentState
vi.mock("#/hooks/use-agent-state");

// Mock useConversationId
vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "test-conversation-id" }),
}));

const createThinkActionEvent = (
  id: string,
  thought: string,
): ActionEvent<ThinkAction> => ({
  id,
  timestamp: new Date().toISOString(),
  source: "agent",
  thought: [
    {
      type: "text",
      text: `think: {"thought": "${thought}"}`,
    },
  ],
  thinking_blocks: [],
  action: {
    kind: "ThinkAction",
    thought,
  },
  tool_name: "think",
  tool_call_id: `call_think_${id}`,
  tool_call: {
    id: `call_think_${id}`,
    type: "function",
    function: {
      name: "think",
      arguments: JSON.stringify({ thought }),
    },
  },
  llm_response_id: `response_${id}`,
  security_risk: SecurityRisk.UNKNOWN,
});

const createBashActionEvent = (
  id: string,
  command: string,
  thoughtText: string,
): ActionEvent<ExecuteBashAction> => ({
  id,
  timestamp: new Date().toISOString(),
  source: "agent",
  thought: [{ type: "text", text: thoughtText }],
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
  security_risk: SecurityRisk.UNKNOWN,
});

describe("EventMessage - ThinkAction rendering", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.INIT,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should NOT render raw tool call text for ThinkAction events", () => {
    const thinkEvent = createThinkActionEvent(
      "think-1",
      "Let me analyze the problem",
    );

    renderWithProviders(
      <EventMessage
        event={thinkEvent}
        messages={[thinkEvent]}
        isLastMessage={false}
        isInLast10Actions={false}
      />,
    );

    // The raw tool call text should NOT be displayed
    expect(
      screen.queryByText(/think: \{"thought":/),
    ).not.toBeInTheDocument();
  });

  it("should render ThinkAction thought as a normal chat message", () => {
    const thinkEvent = createThinkActionEvent(
      "think-2",
      "Let me analyze the problem",
    );

    renderWithProviders(
      <EventMessage
        event={thinkEvent}
        messages={[thinkEvent]}
        isLastMessage={false}
        isInLast10Actions={false}
      />,
    );

    // The thought content should be displayed as regular text
    expect(
      screen.getByText("Let me analyze the problem"),
    ).toBeInTheDocument();

    // It should NOT be inside a collapsible block (no expand button)
    expect(screen.queryByLabelText("Expand")).not.toBeInTheDocument();
  });

  it("should render ThoughtEventMessage for non-ThinkAction events", () => {
    const bashEvent = createBashActionEvent(
      "bash-1",
      "echo hello",
      "I need to run a command",
    );

    renderWithProviders(
      <EventMessage
        event={bashEvent}
        messages={[bashEvent]}
        isLastMessage={false}
        isInLast10Actions={false}
      />,
    );

    // The thought should be displayed for non-think actions
    expect(
      screen.getByText("I need to run a command"),
    ).toBeInTheDocument();
  });
});
