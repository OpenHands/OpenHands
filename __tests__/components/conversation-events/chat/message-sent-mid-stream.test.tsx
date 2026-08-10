import { describe, expect, it } from "vitest";
import { renderWithProviders } from "test-utils";
import { MessageEvent, OpenHandsEvent } from "#/types/agent-server/core";
import { StreamingDeltaEvent } from "#/types/agent-server/core/events/streaming-delta-event";
import { handleEventForUI } from "#/utils/handle-event-for-ui";
import { Messages } from "#/components/conversation-events/chat/messages";
import { shouldRenderEvent } from "#/components/conversation-events/chat/event-content-helpers/should-render-event";

// Regression for #1899, driving the real reducer + <Messages> tree: a message
// sent mid-stream must not split the reply into two bubbles.

const FIRST_HALF = "I am done with the refactor, and ";
const SECOND_HALF = "I also fixed the failing test.";
const REPLY = `${FIRST_HALF}${SECOND_HALF}`;

const countOccurrences = (haystack: string, needle: string): number =>
  haystack.split(needle).length - 1;

const openingMessage: MessageEvent = {
  id: "user-1",
  timestamp: "2026-06-12T12:00:00Z",
  source: "user",
  llm_message: {
    role: "user",
    content: [{ type: "text", text: "Refactor the parser." }],
  },
  activated_microagents: [],
  extended_content: [],
};

const makeDelta = (id: string, content: string): StreamingDeltaEvent => ({
  id,
  kind: "StreamingDeltaEvent",
  timestamp: "2026-06-12T12:00:01Z",
  source: "agent",
  content,
  reasoning_content: null,
});

// Sent while the agent was still streaming, so it arrives between two deltas.
const midStreamMessage: MessageEvent = {
  id: "user-2",
  timestamp: "2026-06-12T12:00:02Z",
  source: "user",
  llm_message: {
    role: "user",
    content: [{ type: "text", text: "also update the README" }],
  },
  activated_microagents: [],
  extended_content: [],
};

const finalMessage: MessageEvent = {
  id: "agent-1",
  timestamp: "2026-06-12T12:00:03Z",
  source: "agent",
  llm_message: { role: "assistant", content: [{ type: "text", text: REPLY }] },
  activated_microagents: [],
  extended_content: [],
};

const reduce = (events: OpenHandsEvent[]): OpenHandsEvent[] =>
  events.reduce<OpenHandsEvent[]>((ui, ev) => handleEventForUI(ev, ui), []);

// Mirrors use-filtered-events.ts: shouldRenderEvent filters the reducer output
// before <Messages>, dropping non-visual events like state snapshots.
const render = (allEvents: OpenHandsEvent[]) =>
  renderWithProviders(
    <Messages
      messages={reduce(allEvents).filter(shouldRenderEvent)}
      allEvents={allEvents}
    />,
  );

// Captured live between the two halves, agent still RUNNING.
const runningStateSnapshot = {
  id: "state-running",
  timestamp: "2026-06-12T12:00:02.5Z",
  source: "environment",
  kind: "ConversationStateUpdateEvent",
  key: "full_state",
  value: { execution_status: "running" },
} as unknown as OpenHandsEvent;

describe("issue #1899 — message sent mid-stream splits the reply", () => {
  const streaming = [
    openingMessage,
    makeDelta("delta-1", FIRST_HALF),
    midStreamMessage,
    runningStateSnapshot,
    makeDelta("delta-2", SECOND_HALF),
  ];

  it("renders the streamed reply as one bubble, not two", () => {
    const { container } = render(streaming);
    const text = container.textContent ?? "";

    // Contiguous reply, above the message. Confirmed live.
    expect(countOccurrences(text, REPLY)).toBe(1);
    expect(text.indexOf(REPLY)).toBeLessThan(text.indexOf("also update"));
  });

  it("renders the reply exactly once when the final message arrives", () => {
    const { container } = render([...streaming, finalMessage]);
    const text = container.textContent ?? "";

    // Final message supersedes both halves — no orphan, no duplication.
    expect(countOccurrences(text, REPLY)).toBe(1);
    expect(countOccurrences(text, SECOND_HALF)).toBe(1);

    // Not asserting reply-above here: the store re-sorts uiEvents by timestamp
    // when the server's trailing snapshots arrive ~ms out of order, settling the
    // reply below the message (also where a reload puts it). See PR notes.
  });
});
