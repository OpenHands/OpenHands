import { describe, expect, it } from "vitest";
import { fireEvent, screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import {
  ActionEvent,
  MessageEvent,
  ObservationEvent,
  OpenHandsEvent,
  SecurityRisk,
} from "#/types/agent-server/core";
import { StreamingDeltaEvent } from "#/types/agent-server/core/events/streaming-delta-event";
import { handleEventForUI } from "#/utils/handle-event-for-ui";
import { Messages } from "#/components/conversation-events/chat/messages";

// Regression for issue #1534. With stream=true the agent streams its
// pre-tool-call text as a StreamingDeltaEvent and then emits an intermediate
// ActionEvent whose `thought` is that same text. The chat hoists the action's
// thought into its own message, so the leftover streaming delta used to render
// the identical text a second time. This test drives the real reducer
// (handleEventForUI) and renders the real <Messages> tree to assert the text
// shows exactly once.

const THOUGHT =
  "I'll create a webpage explaining what OpenHands can do. " +
  "Let me first gather accurate information from the official documentation.";

const countOccurrences = (haystack: string, needle: string): number =>
  haystack.split(needle).length - 1;

const userMessage: MessageEvent = {
  id: "user-1",
  timestamp: "2026-06-12T12:00:00Z",
  source: "user",
  llm_message: {
    role: "user",
    content: [{ type: "text", text: "Create a basic webpage." }],
  },
  activated_microagents: [],
  extended_content: [],
};

// The reasoning streamed as a delta; many models carry reasoning ONLY here, so
// the collapsible "Thinking" must survive the reconciliation.
const streamingDelta: StreamingDeltaEvent = {
  id: "delta-1",
  kind: "StreamingDeltaEvent",
  timestamp: "2026-06-12T12:00:01Z",
  source: "agent",
  content: THOUGHT,
  reasoning_content: "Considering the request before acting.",
};

const action: ActionEvent = {
  id: "action-1",
  timestamp: "2026-06-12T12:00:02Z",
  source: "agent",
  thought: [{ type: "text", text: THOUGHT }],
  thinking_blocks: [],
  action: {
    kind: "ExecuteBashAction",
    command: "ls",
    is_input: false,
    timeout: null,
    reset: false,
  },
  tool_name: "execute_bash",
  tool_call_id: "call_1",
  tool_call: {
    id: "call_1",
    type: "function",
    function: { name: "execute_bash", arguments: '{"command":"ls"}' },
  },
  llm_response_id: "resp-1",
  security_risk: SecurityRisk.UNKNOWN,
};

const observation: ObservationEvent = {
  id: "obs-1",
  timestamp: "2026-06-12T12:00:03Z",
  source: "environment",
  tool_name: "execute_bash",
  tool_call_id: "call_1",
  observation: {
    kind: "ExecuteBashObservation",
    content: [{ type: "text", text: "file.txt\n" }],
    command: "ls",
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
  action_id: "action-1",
};

const reduce = (events: OpenHandsEvent[]): OpenHandsEvent[] =>
  events.reduce<OpenHandsEvent[]>((ui, ev) => handleEventForUI(ev, ui), []);

describe("issue #1534 — streamed intermediate message duplication", () => {
  it("renders the intermediate thought text exactly once", () => {
    const allEvents = [userMessage, streamingDelta, action, observation];
    const uiEvents = reduce(allEvents);

    const { container } = renderWithProviders(
      <Messages messages={uiEvents} allEvents={allEvents} />,
    );

    expect(countOccurrences(container.textContent ?? "", THOUGHT)).toBe(1);
  });

  it("keeps the streamed reasoning visible (Thinking section survives)", () => {
    const allEvents = [userMessage, streamingDelta, action, observation];
    const uiEvents = reduce(allEvents);

    renderWithProviders(<Messages messages={uiEvents} allEvents={allEvents} />);

    // The collapsible "Thinking" section is still rendered (collapsed by
    // default); expanding it reveals the reasoning that was streamed on the
    // delta, proving the reconciliation kept the reasoning.
    fireEvent.click(screen.getByTestId("collapsible-thinking-toggle"));
    expect(
      screen.getByTestId("collapsible-thinking-content").textContent ?? "",
    ).toContain("Considering the request before acting.");
  });

  it("renders a single Thinking section when the action carries its own reasoning", () => {
    // When the finalized action also reports reasoning, the streamed delta's
    // reasoning would duplicate the action's "Thinking" — so the delta is
    // dropped and only one Thinking section renders.
    const reasoningAction: ActionEvent = {
      ...action,
      reasoning_content: "Considering the request before acting.",
    };
    const allEvents = [
      userMessage,
      streamingDelta,
      reasoningAction,
      observation,
    ];
    const uiEvents = reduce(allEvents);

    const { container } = renderWithProviders(
      <Messages messages={uiEvents} allEvents={allEvents} />,
    );

    expect(screen.getAllByTestId("collapsible-thinking")).toHaveLength(1);
    expect(countOccurrences(container.textContent ?? "", THOUGHT)).toBe(1);
  });

  it("renders the canonical final message and its metadata exactly once", () => {
    const finalMessage: MessageEvent = {
      id: "agent-msg-1",
      timestamp: "2026-06-12T12:00:02Z",
      source: "agent",
      llm_message: {
        role: "assistant",
        content: [{ type: "text", text: THOUGHT }],
      },
      activated_microagents: [],
      extended_content: [],
      critic_result: {
        score: 0.72,
        message: null,
        metadata: null,
      },
    };
    const allEvents = [userMessage, streamingDelta, finalMessage];
    const uiEvents = reduce(allEvents);

    expect(uiEvents.at(-1)).toBe(finalMessage);

    const { container } = renderWithProviders(
      <Messages messages={uiEvents} allEvents={allEvents} />,
    );

    expect(countOccurrences(container.textContent ?? "", THOUGHT)).toBe(1);
    expect(screen.getByLabelText("Score: 72.0%")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("collapsible-thinking-toggle"));
    expect(
      screen.getByTestId("collapsible-thinking-content"),
    ).toHaveTextContent("Considering the request before acting.");
  });

  it("renders once when streamed text has leading whitespace", () => {
    const leadingWhitespaceDelta: StreamingDeltaEvent = {
      ...streamingDelta,
      content: `\n${THOUGHT}`,
    };
    const finalMessage: MessageEvent = {
      id: "agent-msg-leading-whitespace",
      timestamp: "2026-06-12T12:00:02Z",
      source: "agent",
      llm_message: {
        role: "assistant",
        content: [{ type: "text", text: THOUGHT }],
      },
      activated_microagents: [],
      extended_content: [],
    };
    const allEvents = [userMessage, leadingWhitespaceDelta, finalMessage];
    const uiEvents = reduce(allEvents);

    expect(uiEvents.at(-1)).toBe(finalMessage);

    const { container } = renderWithProviders(
      <Messages messages={uiEvents} allEvents={allEvents} />,
    );

    expect(countOccurrences(container.textContent ?? "", THOUGHT)).toBe(1);
  });
});

// A model that streams its reasoning but whose action `thought` does not repeat
// that text used to render the delta's reasoning and the action's own reasoning
// as two identical "Thinking" bubbles.
describe("duplicate Thinking blocks", () => {
  const REASONING =
    "Let me read the full PRD to understand all the requirements. " +
    "I need to see sections that were clipped.";

  const reasoningOnlyDelta: StreamingDeltaEvent = {
    id: "delta-reasoning-only",
    kind: "StreamingDeltaEvent",
    timestamp: "2026-06-12T12:00:01Z",
    source: "agent",
    content: null,
    reasoning_content: REASONING,
  };

  const reasoningAction: ActionEvent = {
    ...action,
    thought: [],
    reasoning_content: REASONING,
  };

  it("renders the reasoning in a single Thinking section", () => {
    const allEvents = [
      userMessage,
      reasoningOnlyDelta,
      reasoningAction,
      observation,
    ];
    const uiEvents = reduce(allEvents);

    const { container } = renderWithProviders(
      <Messages messages={uiEvents} allEvents={allEvents} />,
    );

    expect(screen.getAllByTestId("collapsible-thinking")).toHaveLength(1);

    fireEvent.click(screen.getByTestId("collapsible-thinking-toggle"));
    expect(
      screen.getByTestId("collapsible-thinking-content"),
    ).toHaveTextContent(REASONING);
    expect(countOccurrences(container.textContent ?? "", REASONING)).toBe(1);
  });

  it("keeps the streamed reasoning when the action carries none", () => {
    const plainAction: ActionEvent = { ...action, thought: [] };
    const allEvents = [
      userMessage,
      reasoningOnlyDelta,
      plainAction,
      observation,
    ];
    const uiEvents = reduce(allEvents);

    renderWithProviders(<Messages messages={uiEvents} allEvents={allEvents} />);

    // The delta is the sole reasoning carrier, so its Thinking must survive.
    expect(screen.getAllByTestId("collapsible-thinking")).toHaveLength(1);
    fireEvent.click(screen.getByTestId("collapsible-thinking-toggle"));
    expect(
      screen.getByTestId("collapsible-thinking-content"),
    ).toHaveTextContent(REASONING);
  });
});
