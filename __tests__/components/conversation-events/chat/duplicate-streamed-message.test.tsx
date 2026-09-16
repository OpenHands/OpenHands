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
import {
  appendStreamingDeltas,
  handleEventForUI,
  openStreamingSlot,
} from "#/utils/handle-event-for-ui";
import { Messages } from "#/components/conversation-events/chat/messages";

// Regression for issue #1534. With stream=true the agent streams its
// pre-tool-call text into a slot and then emits the intermediate ActionEvent
// whose `thought` is that same text — built with the slot's own id, because
// the SDK mints it when the stream opens. The chat hoists the action's thought
// into its own message, so the slot must be retired rather than left to render
// the identical text a second time. These tests drive the real reducers and
// render the real <Messages> tree to assert the text shows exactly once.

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
  activated_skills: [],
  extended_content: [],
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

/**
 * Replay a turn: `stream(itemId, …)` opens a slot and feeds it, everything
 * else goes through the durable reducer — the same order the session socket
 * delivers them in.
 */
const stream = (
  itemId: string,
  text: string | null,
  reasoning: string | null = null,
) => ({ itemId, text, reasoning });

type Step = OpenHandsEvent | ReturnType<typeof stream>;

const isStreamStep = (step: Step): step is ReturnType<typeof stream> =>
  "itemId" in step;

/** The durable events of a replay, in arrival order (the `allEvents` list). */
const durableOf = (steps: Step[]): OpenHandsEvent[] =>
  steps.filter((step): step is OpenHandsEvent => !isStreamStep(step));

const replay = (steps: Step[]): OpenHandsEvent[] =>
  steps.reduce<OpenHandsEvent[]>((ui, step) => {
    if (!isStreamStep(step)) {
      return handleEventForUI(step, ui);
    }
    const opened = openStreamingSlot(
      { type: "item_started", item_id: step.itemId },
      ui,
    );
    return appendStreamingDeltas(
      [
        ...(step.reasoning
          ? [
              {
                type: "delta" as const,
                item_id: step.itemId,
                attempt: 1,
                order: 0,
                kind: "reasoning" as const,
                content: step.reasoning,
              },
            ]
          : []),
        ...(step.text
          ? [
              {
                type: "delta" as const,
                item_id: step.itemId,
                attempt: 1,
                order: 1,
                kind: "text" as const,
                content: step.text,
              },
            ]
          : []),
      ],
      opened,
    );
  }, []);

describe("issue #1534 — streamed intermediate message duplication", () => {
  it("renders the intermediate thought text exactly once", () => {
    const steps: Step[] = [
      userMessage,
      stream(action.id, THOUGHT, "Considering the request before acting."),
      action,
      observation,
    ];
    const uiEvents = replay(steps);

    const { container } = renderWithProviders(
      <Messages messages={uiEvents} allEvents={durableOf(steps)} />,
    );

    expect(countOccurrences(container.textContent ?? "", THOUGHT)).toBe(1);
  });

  it("retires the slot on the action that carries its id", () => {
    const streamed = replay([userMessage, stream(action.id, THOUGHT)]);
    // While streaming, the slot renders the provisional text.
    expect(streamed.map((event) => event.id)).toEqual([
      userMessage.id,
      action.id,
    ]);

    const uiEvents = handleEventForUI(action, streamed);
    expect(uiEvents).toEqual([userMessage, action]);
  });

  it("renders a single Thinking section when the action carries its own reasoning", () => {
    const reasoningAction: ActionEvent = {
      ...action,
      reasoning_content: "Considering the request before acting.",
    };
    const steps: Step[] = [
      userMessage,
      stream(action.id, THOUGHT, "Considering the request before acting."),
      reasoningAction,
      observation,
    ];
    const uiEvents = replay(steps);

    const { container } = renderWithProviders(
      <Messages messages={uiEvents} allEvents={durableOf(steps)} />,
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
      activated_skills: [],
      extended_content: [],
      critic_result: {
        score: 0.72,
        message: null,
        metadata: null,
      },
    };
    const steps: Step[] = [
      userMessage,
      stream(finalMessage.id, THOUGHT),
      finalMessage,
    ];
    const uiEvents = replay(steps);

    expect(uiEvents.at(-1)).toBe(finalMessage);

    const { container } = renderWithProviders(
      <Messages messages={uiEvents} allEvents={durableOf(steps)} />,
    );

    expect(countOccurrences(container.textContent ?? "", THOUGHT)).toBe(1);
    expect(screen.getByLabelText("Score: 72.0%")).toBeInTheDocument();
  });

  it("keeps the streamed reasoning visible while the slot is still open", () => {
    const steps: Step[] = [
      userMessage,
      stream(action.id, THOUGHT, "Considering the request before acting."),
    ];
    const uiEvents = replay(steps);

    renderWithProviders(
      <Messages messages={uiEvents} allEvents={durableOf(steps)} />,
    );

    fireEvent.click(screen.getByTestId("collapsible-thinking-toggle"));
    expect(
      screen.getByTestId("collapsible-thinking-content").textContent ?? "",
    ).toContain("Considering the request before acting.");
  });
});

// A model that streams reasoning only: the slot renders it until the durable
// action — which carries the same reasoning, and the slot's id — replaces it.
describe("duplicate Thinking blocks", () => {
  const REASONING =
    "Let me read the full PRD to understand all the requirements. " +
    "I need to see sections that were clipped.";

  const reasoningAction: ActionEvent = {
    ...action,
    thought: [],
    reasoning_content: REASONING,
  };

  it("renders the reasoning in a single Thinking section", () => {
    const steps: Step[] = [
      userMessage,
      stream(reasoningAction.id, null, REASONING),
      reasoningAction,
      observation,
    ];
    const uiEvents = replay(steps);

    const { container } = renderWithProviders(
      <Messages messages={uiEvents} allEvents={durableOf(steps)} />,
    );

    expect(screen.getAllByTestId("collapsible-thinking")).toHaveLength(1);

    fireEvent.click(screen.getByTestId("collapsible-thinking-toggle"));
    expect(
      screen.getByTestId("collapsible-thinking-content"),
    ).toHaveTextContent(REASONING);
    expect(countOccurrences(container.textContent ?? "", REASONING)).toBe(1);
  });

  it("renders the streamed reasoning while the slot is open", () => {
    const steps: Step[] = [
      userMessage,
      stream(reasoningAction.id, null, REASONING),
    ];
    const uiEvents = replay(steps);

    renderWithProviders(
      <Messages messages={uiEvents} allEvents={durableOf(steps)} />,
    );

    expect(screen.getAllByTestId("collapsible-thinking")).toHaveLength(1);
    fireEvent.click(screen.getByTestId("collapsible-thinking-toggle"));
    expect(
      screen.getByTestId("collapsible-thinking-content"),
    ).toHaveTextContent(REASONING);
  });
});
