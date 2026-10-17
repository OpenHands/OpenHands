import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { EventMessage } from "#/components/conversation-events/chat/event-message";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";
import { ActionEvent, SecurityRisk } from "#/types/agent-server/core";
import {
  ThinkAction,
  ExecuteBashAction,
  FinishAction,
} from "#/types/agent-server/core/base/action";
import { renderWithProviders } from "test-utils";

// Mock useConfig
vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({
    data: {},
  }),
}));

// Mock useAgentState
vi.mock("#/hooks/use-agent-state");

// Mock useConversationId
vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "test-conversation-id" }),
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
  overrides?: Partial<ActionEvent<ExecuteBashAction>>,
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
  ...overrides,
});

const createFinishActionEvent = (
  id: string,
  message: string,
  overrides?: Partial<ActionEvent<FinishAction>>,
): ActionEvent<FinishAction> => ({
  id,
  timestamp: new Date().toISOString(),
  source: "agent",
  thought: [],
  thinking_blocks: [],
  action: {
    kind: "FinishAction",
    message,
  },
  tool_name: "finish",
  tool_call_id: `call_finish_${id}`,
  tool_call: {
    id: `call_finish_${id}`,
    type: "function",
    function: {
      name: "finish",
      arguments: JSON.stringify({ message }),
    },
  },
  llm_response_id: `response_${id}`,
  security_risk: SecurityRisk.UNKNOWN,
  ...overrides,
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
    expect(screen.queryByText(/think: \{"thought":/)).not.toBeInTheDocument();
  });

  it("should render ThinkAction as a collapsible section", () => {
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

    // The collapsible thinking wrapper should exist
    expect(screen.getByTestId("collapsible-thinking")).toBeInTheDocument();

    // The thought content should NOT be visible initially (collapsed by default)
    expect(
      screen.queryByTestId("collapsible-thinking-content"),
    ).not.toBeInTheDocument();
  });

  it("should expand ThinkAction content when toggle is clicked", async () => {
    const user = userEvent.setup();
    const thinkEvent = createThinkActionEvent(
      "think-3",
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

    // Click the toggle to expand
    await user.click(screen.getByTestId("collapsible-thinking-toggle"));

    // Now the content should be visible
    expect(
      screen.getByTestId("collapsible-thinking-content"),
    ).toBeInTheDocument();
    expect(screen.getByText("Let me analyze the problem")).toBeInTheDocument();
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
    expect(screen.getByText("I need to run a command")).toBeInTheDocument();
  });

  it("should render reasoning_content as a collapsible section", () => {
    const bashEvent = createBashActionEvent(
      "bash-reasoning",
      "echo hello",
      "Running a command",
      {
        reasoning_content: "I need to think carefully about this step.",
      },
    );

    renderWithProviders(
      <EventMessage
        event={bashEvent}
        messages={[bashEvent]}
        isLastMessage={false}
        isInLast10Actions={false}
      />,
    );

    // The collapsible thinking wrapper should exist for reasoning_content
    expect(screen.getByTestId("collapsible-thinking")).toBeInTheDocument();

    // The reasoning content should be hidden initially
    expect(
      screen.queryByText("I need to think carefully about this step."),
    ).not.toBeInTheDocument();
  });

  it("should render thinking_blocks as a collapsible section", () => {
    const bashEvent = createBashActionEvent(
      "bash-thinking-blocks",
      "echo hello",
      "Running a command",
      {
        thinking_blocks: [
          {
            type: "thinking",
            thinking: "Extended thinking block content here.",
            signature: "sig123",
          },
        ],
      },
    );

    renderWithProviders(
      <EventMessage
        event={bashEvent}
        messages={[bashEvent]}
        isLastMessage={false}
        isInLast10Actions={false}
      />,
    );

    // The collapsible thinking wrapper should exist for thinking_blocks
    expect(screen.getByTestId("collapsible-thinking")).toBeInTheDocument();
  });

  describe("FinishAction reasoning rendering", () => {
    it("should render reasoning_content on FinishAction as a collapsible section", async () => {
      const user = userEvent.setup();
      const finishEvent = createFinishActionEvent(
        "finish-reasoning",
        "Task completed successfully.",
        {
          reasoning_content: "Reviewed test suite and verified all cases pass.",
        },
      );

      renderWithProviders(
        <EventMessage
          event={finishEvent}
          messages={[finishEvent]}
          isLastMessage={true}
          isInLast10Actions={true}
        />,
      );

      // The collapsible thinking section should exist and be collapsed by default
      expect(screen.getByTestId("collapsible-thinking")).toBeInTheDocument();
      expect(
        screen.queryByTestId("collapsible-thinking-content"),
      ).not.toBeInTheDocument();
      expect(
        screen.getByText("Task completed successfully."),
      ).toBeInTheDocument();

      // Expand thinking section
      await user.click(screen.getByTestId("collapsible-thinking-toggle"));
      expect(
        screen.getByTestId("collapsible-thinking-content"),
      ).toBeInTheDocument();
      expect(
        screen.getByText("Reviewed test suite and verified all cases pass."),
      ).toBeInTheDocument();
    });

    it("should NOT render collapsible thinking section when FinishAction has no reasoning", () => {
      const finishEvent = createFinishActionEvent(
        "finish-no-reasoning",
        "All done without thinking.",
      );

      renderWithProviders(
        <EventMessage
          event={finishEvent}
          messages={[finishEvent]}
          isLastMessage={true}
          isInLast10Actions={true}
        />,
      );

      expect(
        screen.queryByTestId("collapsible-thinking"),
      ).not.toBeInTheDocument();
      expect(
        screen.getByText("All done without thinking."),
      ).toBeInTheDocument();
    });

    it("should NOT render collapsible thinking section when FinishAction has whitespace-only reasoning", () => {
      const finishEvent = createFinishActionEvent(
        "finish-whitespace-reasoning",
        "Done with empty reasoning.",
        {
          reasoning_content: "   \n\t  ",
        },
      );

      renderWithProviders(
        <EventMessage
          event={finishEvent}
          messages={[finishEvent]}
          isLastMessage={true}
          isInLast10Actions={true}
        />,
      );

      expect(
        screen.queryByTestId("collapsible-thinking"),
      ).not.toBeInTheDocument();
      expect(
        screen.getByText("Done with empty reasoning."),
      ).toBeInTheDocument();
    });

    it("should render thinking_blocks on FinishAction as a collapsible section", async () => {
      const user = userEvent.setup();
      const finishEvent = createFinishActionEvent(
        "finish-thinking-blocks",
        "Done with thinking blocks.",
        {
          thinking_blocks: [
            {
              type: "thinking",
              thinking: "Anthropic finish reasoning block.",
              signature: "sig456",
            },
          ],
        },
      );

      renderWithProviders(
        <EventMessage
          event={finishEvent}
          messages={[finishEvent]}
          isLastMessage={true}
          isInLast10Actions={true}
        />,
      );

      expect(screen.getByTestId("collapsible-thinking")).toBeInTheDocument();
      await user.click(screen.getByTestId("collapsible-thinking-toggle"));
      expect(
        screen.getByText("Anthropic finish reasoning block."),
      ).toBeInTheDocument();
    });

    it("should keep assistant message unmodified when FinishAction has no reasoning_content", () => {
      const finishEvent = createFinishActionEvent(
        "finish-raw-message",
        "ACP completed: done with all steps.",
      );

      renderWithProviders(
        <EventMessage
          event={finishEvent}
          messages={[finishEvent]}
          isLastMessage={true}
          isInLast10Actions={true}
        />,
      );

      expect(
        screen.queryByTestId("collapsible-thinking"),
      ).not.toBeInTheDocument();
      expect(
        screen.getByText("ACP completed: done with all steps."),
      ).toBeInTheDocument();
    });
  });
});
