import { describe, expect, it } from "vitest";
import { indexDecisionClosures } from "#/utils/decision-closures";
import type { OHEvent } from "#/stores/use-event-store";

const COORDINATOR = "conversation:cc000000000040008000000000000001";

const question = (id: string) =>
  ({
    id,
    timestamp: "2026-11-16T12:00:01Z",
    source: "environment",
    tool_name: "ask_user",
    tool_call_id: `call-${id}`,
    action_id: `action-${id}`,
    observation: {
      kind: "AskUserObservation",
      content: [{ type: "text", text: "The question is shown." }],
      is_error: false,
    },
  }) as unknown as OHEvent;

const answer = (id: string, askId: string) =>
  ({
    id,
    kind: "AskUserAnswerEvent",
    timestamp: "2026-11-16T12:00:02Z",
    source: "user",
    ask_id: askId,
    option_index: 0,
    option_label: "SQLite",
    text: null,
  }) as unknown as OHEvent;

const message = (id: string, sender?: string) =>
  ({
    id,
    timestamp: "2026-11-16T12:00:03Z",
    source: "user",
    sender,
    llm_message: {
      role: "user",
      content: [{ type: "text", text: '[Answer to your question: "?"]' }],
    },
    activated_skills: [],
    extended_content: [],
  }) as unknown as OHEvent;

describe("indexDecisionClosures in a project", () => {
  it("closes a worker's question when the coordinator delivers the answer", () => {
    const closures = indexDecisionClosures([
      question("q1"),
      answer("a1", "action-q1"),
      message("delivery", COORDINATOR),
    ]);
    expect(closures.get("q1")?.closing.id).toBe("a1");
    expect(closures.get("q1")?.delivery?.id).toBe("delivery");
  });

  it("keeps an answer that was never delivered retryable", () => {
    const closures = indexDecisionClosures([
      question("q1"),
      answer("a1", "action-q1"),
    ]);
    expect(closures.has("q1")).toBe(false);
  });

  it("does not let another conversation's message alone close a question", () => {
    const closures = indexDecisionClosures([
      question("q1"),
      message("relayed", COORDINATOR),
    ]);
    expect(closures.has("q1")).toBe(false);
  });

  it("does not count a delivery that came after the next question", () => {
    const closures = indexDecisionClosures([
      question("q1"),
      answer("a1", "action-q1"),
      question("q2"),
      message("delivery", COORDINATOR),
    ]);
    expect(closures.get("q1")?.closing.id).toBe("q2");
    expect(closures.has("q2")).toBe(false);
  });
});
