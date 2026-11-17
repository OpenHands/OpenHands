import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import {
  appendStreamingDeltas,
  handleEventForUI,
  openStreamingSlot,
} from "#/utils/handle-event-for-ui";
import { Messages } from "#/components/conversation-events/chat/messages";
import { MessageEvent, OpenHandsEvent } from "#/types/agent-server/core";

// Regression for the memo comparator on <Messages>: its checks were all
// append-only (lengths + last element identity). When a user message lands
// below an open streaming slot (#15433 insertion), the slot sits mid-list and
// every later delta batch replaces it IN PLACE — same length, same last
// element — so the comparator passed and the streamed bubble froze until the
// durable event retired the slot.

const userBefore: MessageEvent = {
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

const userInterjection: MessageEvent = {
  id: "user-2",
  timestamp: "2026-06-12T12:00:01Z",
  source: "user",
  llm_message: {
    role: "user",
    content: [{ type: "text", text: "Also add a footer." }],
  },
  activated_skills: [],
  extended_content: [],
};

const stream = (itemId: string, text: string) => ({ itemId, text });

type Step = OpenHandsEvent | ReturnType<typeof stream>;

const isStreamStep = (step: Step): step is ReturnType<typeof stream> =>
  "itemId" in step;

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
        {
          type: "delta" as const,
          item_id: step.itemId,
          attempt: 1,
          order: 0,
          kind: "text" as const,
          content: step.text ?? "",
        },
      ],
      opened,
    );
  }, []);

describe("mid-list streaming slot updates", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("keeps the streamed bubble updating when a user message lands below the slot", () => {
    // The rAF reveal parks the bubble at the old text when nothing ticks, so
    // stub it off: useStreamedText then commits synchronously and this test
    // measures render propagation (the memo comparator), not the animation.
    vi.stubGlobal("requestAnimationFrame", undefined);

    // Phase 1: the agent streams partial text into an open slot and the user
    // interjection lands below it — the slot sits mid-list from here on.
    let ui = replay([userBefore, stream("call_1", "Hel"), userInterjection]);

    const view = renderWithProviders(<Messages messages={ui} allEvents={[]} />);
    expect(view.container.textContent).toContain("Hel");

    // Phase 2: the next delta batch replaces the slot IN PLACE, mid-list —
    // same messages length, same neighbouring and last element references.
    // The comparator must notice the slot's new identity and re-render
    // instead of freezing the bubble at the partial text.
    ui = replay([userBefore, stream("call_1", "Hello"), userInterjection]);
    view.rerender(<Messages messages={ui} allEvents={[]} />);

    expect(view.container.textContent).toContain("Hello");
    // "Hello" contains "Hel", so a plain not.toContain would always fail;
    // assert no stale "Hel" survives outside the full word.
    expect(view.container.textContent).not.toMatch(/Hel(?!lo)/);
  });
});
