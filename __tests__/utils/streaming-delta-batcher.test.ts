import { describe, it, expect } from "vitest";
import {
  createStreamingDeltaBatcher,
  DeltaFlushScheduler,
} from "#/utils/streaming-delta-batcher";
import { useEventStore } from "#/stores/use-event-store";
import type { DeltaFrame } from "#/types/agent-server/session-frames";
import { MessageEvent } from "#/types/agent-server/core";
import { isStreamingDeltaEvent } from "#/types/agent-server/type-guards";

const ITEM_ID = "agent-1";

const makeDelta = (
  order: number,
  content: string,
  kind: DeltaFrame["kind"] = "text",
  itemId = ITEM_ID,
): DeltaFrame => ({
  type: "delta",
  item_id: itemId,
  attempt: 1,
  order,
  kind,
  content,
});

/**
 * Deterministic stand-in for `requestAnimationFrame`: callbacks only run when
 * the test explicitly `tick()`s a frame, so cadence is fully controlled.
 */
function manualScheduler() {
  const callbacks = new Map<number, () => void>();
  let nextHandle = 1;
  const scheduler: DeltaFlushScheduler = {
    schedule: (callback) => {
      const handle = nextHandle;
      nextHandle += 1;
      callbacks.set(handle, callback);
      return handle;
    },
    cancel: (handle) => {
      callbacks.delete(handle);
    },
  };
  return {
    scheduler,
    pendingFrames: () => callbacks.size,
    tick: () => {
      const scheduled = [...callbacks.values()];
      callbacks.clear();
      scheduled.forEach((callback) => callback());
    },
  };
}

describe("createStreamingDeltaBatcher", () => {
  it("coalesces adjacent deltas into a single commit per frame", () => {
    const commits: DeltaFrame[][] = [];
    const clock = manualScheduler();
    const batcher = createStreamingDeltaBatcher(
      (frames) => commits.push(frames),
      clock.scheduler,
    );

    batcher.enqueue(makeDelta(0, "Hello"));
    batcher.enqueue(makeDelta(1, ", "));
    batcher.enqueue(makeDelta(2, "world"));
    expect(commits).toHaveLength(0);
    expect(clock.pendingFrames()).toBe(1);

    clock.tick();
    expect(commits).toHaveLength(1);
    expect(commits[0].map((frame) => frame.content).join("")).toBe(
      "Hello, world",
    );
  });

  it("keeps each frame's kind and order intact for the store to apply", () => {
    const commits: DeltaFrame[][] = [];
    const clock = manualScheduler();
    const batcher = createStreamingDeltaBatcher(
      (frames) => commits.push(frames),
      clock.scheduler,
    );

    batcher.enqueue(makeDelta(0, "think-", "reasoning"));
    batcher.enqueue(makeDelta(1, "ans"));
    batcher.enqueue(makeDelta(2, "more", "reasoning"));
    clock.tick();

    expect(commits[0].map((frame) => [frame.kind, frame.content])).toEqual([
      ["reasoning", "think-"],
      ["text", "ans"],
      ["reasoning", "more"],
    ]);
  });

  it("flush() commits synchronously and cancels the scheduled frame", () => {
    const commits: DeltaFrame[][] = [];
    const clock = manualScheduler();
    const batcher = createStreamingDeltaBatcher(
      (frames) => commits.push(frames),
      clock.scheduler,
    );

    batcher.enqueue(makeDelta(0, "a"));
    batcher.enqueue(makeDelta(1, "b"));
    batcher.flush();

    expect(commits).toHaveLength(1);
    expect(clock.pendingFrames()).toBe(0);

    // Nothing buffered: a second flush is a no-op.
    batcher.flush();
    expect(commits).toHaveLength(1);
  });

  it("reset() drops buffered deltas without committing them", () => {
    const commits: DeltaFrame[][] = [];
    const clock = manualScheduler();
    const batcher = createStreamingDeltaBatcher(
      (frames) => commits.push(frames),
      clock.scheduler,
    );

    batcher.enqueue(makeDelta(0, "lost"));
    batcher.reset();
    clock.tick();

    expect(commits).toHaveLength(0);
    expect(clock.pendingFrames()).toBe(0);
  });

  it("preserves text byte-for-byte and order across thousands of 1-char deltas faster than 60Hz", () => {
    const commits: DeltaFrame[][] = [];
    const clock = manualScheduler();
    const batcher = createStreamingDeltaBatcher(
      (frames) => commits.push(frames),
      clock.scheduler,
    );

    const total = 5000;
    let expected = "";
    for (let i = 0; i < total; i += 1) {
      const char = String.fromCharCode(97 + (i % 26));
      expected += char;
      batcher.enqueue(makeDelta(i, char));
      // A frame only every 100 deltas => deltas arrive far faster than frames.
      if (i % 100 === 99) {
        clock.tick();
      }
    }
    batcher.flush(); // boundary flush, as any non-delta frame would trigger

    // Commits are bounded by frames, not by provider chunk count.
    expect(commits.length).toBeLessThan(total);
    expect(commits.length).toBeLessThanOrEqual(total / 100 + 1);
    expect(
      commits
        .flat()
        .map((frame) => frame.content)
        .join(""),
    ).toBe(expected);
  });
});

describe("createStreamingDeltaBatcher wired into the event store", () => {
  const userMessage: MessageEvent = {
    id: "user-1",
    timestamp: "2024-02-01T00:00:00Z",
    source: "user",
    llm_message: { role: "user", content: [{ type: "text", text: "hi" }] },
    activated_skills: [],
    extended_content: [],
  };

  it("coalesces deltas across frames, then retires the slot when the durable message arrives", () => {
    useEventStore.getState().clearEvents();
    const clock = manualScheduler();
    // Commit into the real store exactly as ConversationWebSocketProvider does.
    const batcher = createStreamingDeltaBatcher(
      (frames) => useEventStore.getState().appendStreamingDeltas(frames),
      clock.scheduler,
    );

    useEventStore.getState().addEvent(userMessage);
    useEventStore
      .getState()
      .openStreamingSlot({ type: "item_started", item_id: ITEM_ID });

    // Stream one char per delta, flushing a frame only every 5 chars, so deltas
    // arrive faster than frames — the case where the UI used to fall behind.
    const streamed = "I'll start working on that.";
    [...streamed].forEach((char, i) => {
      batcher.enqueue(makeDelta(i, char));
      if (i % 5 === 4) {
        clock.tick();
      }
    });
    batcher.flush();

    const slot = useEventStore
      .getState()
      .uiEvents.find((event) => isStreamingDeltaEvent(event));
    expect(slot?.content).toBe(streamed);

    // The durable message carries the slot's id, so it retires it outright.
    const finalMessage: MessageEvent = {
      id: ITEM_ID,
      timestamp: "2024-04-01T00:00:00Z",
      source: "agent",
      llm_message: {
        role: "assistant",
        content: [{ type: "text", text: "I'll start working on that. Done." }],
      },
      activated_skills: [],
      extended_content: [],
    };
    useEventStore.getState().addEvent(finalMessage);

    const state = useEventStore.getState();
    expect(state.uiEvents).toHaveLength(2);
    const bubble = state.uiEvents[1] as MessageEvent;
    expect(bubble.id).toBe(ITEM_ID);
    expect(bubble.llm_message.content).toEqual([
      { type: "text", text: "I'll start working on that. Done." },
    ]);
    // No provisional slot survives, so the streamed text renders exactly once.
    expect(state.uiEvents.some((event) => isStreamingDeltaEvent(event))).toBe(
      false,
    );
    // eventIds tracks only the two durable events, never the 27 deltas.
    expect(state.eventIds.size).toBe(2);
  });
});
