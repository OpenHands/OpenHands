import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { EventMessage } from "#/components/conversation-events/chat/event-message";
import { Messages } from "#/components/conversation-events/chat/messages";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";
import {
  ActionEvent,
  ObservationEvent,
  SecurityRisk,
} from "#/types/agent-server/core";
import {
  ThinkAction,
  ExecuteBashAction,
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

const createBashObservationEvent = (
  id: string,
  actionId: string,
): ObservationEvent => ({
  id,
  timestamp: new Date().toISOString(),
  source: "environment",
  tool_name: "execute_bash",
  tool_call_id: `call_bash_${actionId}`,
  observation: {
    kind: "ExecuteBashObservation",
    content: [{ type: "text", text: "ok\n" }],
    command: "echo hello",
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
  action_id: actionId,
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

  // Regression: a model that emits its reasoning inline in the thought (instead
  // of via reasoning_content) leaked the raw reasoning block into the bubble.
  it("routes an inline reasoning block in an action thought to the thinking section", async () => {
    const user = userEvent.setup();
    const bashEvent = createBashActionEvent(
      "bash-inline-think",
      "echo hello",
      `<think>Let me check the working directory first.</think>\nRunning the command now.`,
    );

    renderWithProviders(
      <EventMessage
        event={bashEvent}
        messages={[bashEvent]}
        isLastMessage={false}
        isInLast10Actions={false}
      />,
    );

    // Reasoning moves to exactly one collapsible section, collapsed by default
    expect(screen.getAllByTestId("collapsible-thinking")).toHaveLength(1);
    expect(
      screen.queryByText("Let me check the working directory first."),
    ).not.toBeInTheDocument();

    // The bubble keeps only the non-reasoning thought, with no raw tags
    expect(screen.getByText("Running the command now.")).toBeInTheDocument();
    expect(screen.getByTestId("agent-message").textContent).not.toContain(
      "Let me check the working directory first.",
    );

    await user.click(screen.getByTestId("collapsible-thinking-toggle"));
    expect(
      screen.getByText("Let me check the working directory first."),
    ).toBeInTheDocument();
  });

  it("renders no bubble when an action thought is inline reasoning only", () => {
    const bashEvent = createBashActionEvent(
      "bash-inline-think-only",
      "echo hello",
      `<think>Just thinking about the command.</think>`,
    );

    renderWithProviders(
      <EventMessage
        event={bashEvent}
        messages={[bashEvent]}
        isLastMessage={false}
        isInLast10Actions={false}
      />,
    );

    expect(screen.getByTestId("collapsible-thinking")).toBeInTheDocument();
    expect(screen.queryByTestId("agent-message")).not.toBeInTheDocument();
  });

  it("renders the inline reasoning once when the paired observation row is shown", () => {
    const bashEvent = createBashActionEvent(
      "bash-inline-think-obs",
      "echo hello",
      `<think>Let me check the working directory first.</think>\nRunning the command now.`,
    );
    const observation = createBashObservationEvent(
      "obs-inline-think",
      bashEvent.id,
    );

    renderWithProviders(
      <EventMessage
        event={observation}
        messages={[bashEvent, observation]}
        isLastMessage={false}
        isInLast10Actions={false}
      />,
    );

    // The thought is rendered through the observation row only — once.
    expect(screen.getAllByTestId("collapsible-thinking")).toHaveLength(1);
    expect(
      screen.queryByText("Let me check the working directory first."),
    ).not.toBeInTheDocument();
    expect(screen.getByText("Running the command now.")).toBeInTheDocument();
  });

  it("does not treat a non-leading reasoning block as reasoning", () => {
    const bashEvent = createBashActionEvent(
      "bash-mid-inline-think",
      "echo hello",
      `Quoting <think>literal</think> tags stays visible.`,
    );

    renderWithProviders(
      <EventMessage
        event={bashEvent}
        messages={[bashEvent]}
        isLastMessage={false}
        isInLast10Actions={false}
      />,
    );

    // Only a leading block is reasoning; a later occurrence stays in the bubble.
    expect(
      screen.queryByTestId("collapsible-thinking"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("agent-message").textContent).toContain(
      "Quoting",
    );
    expect(screen.getByTestId("agent-message").textContent).toContain(
      "tags stays visible.",
    );
  });

  it("keeps an action thought with no inline reasoning in the bubble", () => {
    const bashEvent = createBashActionEvent(
      "bash-no-inline-think",
      "echo hello",
      "Plain thought with no reasoning block.",
    );

    renderWithProviders(
      <EventMessage
        event={bashEvent}
        messages={[bashEvent]}
        isLastMessage={false}
        isInLast10Actions={false}
      />,
    );

    expect(
      screen.getByText("Plain thought with no reasoning block."),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("collapsible-thinking"),
    ).not.toBeInTheDocument();
  });
});

describe("EventMessage - single thinking section per action", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.INIT,
    });
  });

  // Regression (reviewer): an action carrying both reasoning_content and an
  // identical inline block rendered two thinking controls with repeated text.
  it("renders one thinking section when explicit and inline reasoning match", async () => {
    const user = userEvent.setup();
    const bashEvent = createBashActionEvent(
      "bash-dup-reasoning",
      "echo hello",
      "<think>Check the directory</think>\nRunning ls",
      { reasoning_content: "Check the directory" },
    );

    renderWithProviders(
      <Messages messages={[bashEvent]} allEvents={[bashEvent]} />,
    );

    expect(screen.getAllByTestId("collapsible-thinking")).toHaveLength(1);
    expect(screen.getByText("Running ls")).toBeInTheDocument();

    await user.click(screen.getByTestId("collapsible-thinking-toggle"));
    const content = screen.getByTestId("collapsible-thinking-content");
    expect(content.textContent?.match(/Check the directory/g)).toHaveLength(1);
  });

  it("renders one thinking section for distinct explicit and inline reasoning", () => {
    const bashEvent = createBashActionEvent(
      "bash-distinct-reasoning",
      "echo hello",
      "<think>Inline reasoning</think>\nRunning ls",
      { reasoning_content: "Explicit reasoning" },
    );

    renderWithProviders(
      <Messages messages={[bashEvent]} allEvents={[bashEvent]} />,
    );

    expect(screen.getAllByTestId("collapsible-thinking")).toHaveLength(1);
    expect(screen.getByText("Running ls")).toBeInTheDocument();
  });
});
