import { describe, it, expect } from "vitest";
import {
  createStreamingDeltaBatcher,
  DeltaFlushScheduler,
} from "#/utils/streaming-delta-batcher";
import { useEventStore } from "#/stores/use-event-store";
import { StreamingDeltaEvent } from "#/types/agent-server/core/events/streaming-delta-event";
import { MessageEvent } from "#/types/agent-server/core";
import { isStreamingDeltaEvent } from "#/types/agent-server/type-guards";

const makeDelta = (
  id: string,
  content: string | null,
  reasoning: string | null = null,
): StreamingDeltaEvent => ({
  id,
  timestamp: "2024-03-01T00:00:00Z",
  source: "agent",
  kind: "StreamingDeltaEvent",
  content,
  reasoning_content: reasoning,
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
    const commits: StreamingDeltaEvent[] = [];
    const clock = manualScheduler();
    const batcher = createStreamingDeltaBatcher(
      (delta) => commits.push(delta),
      clock.scheduler,
    );

    batcher.enqueue(makeDelta("d1", "Hello"));
    batcher.enqueue(makeDelta("d2", ", "));
    batcher.enqueue(makeDelta("d3", "world"));

    // Nothing commits until the frame fires, and three enqueues schedule only
    // ONE frame (not one per delta).
    expect(commits).toHaveLength(0);
    expect(clock.pendingFrames()).toBe(1);

    clock.tick();

    expect(commits).toHaveLength(1);
    expect(commits[0].content).toBe("Hello, world");
    // The coalesced event keeps the first delta's identity.
    expect(commits[0].id).toBe("d1");
  });

  it("merges content and reasoning_content independently, in order", () => {
    const commits: StreamingDeltaEvent[] = [];
    const clock = manualScheduler();
    const batcher = createStreamingDeltaBatcher(
      (delta) => commits.push(delta),
      clock.scheduler,
    );

    batcher.enqueue(makeDelta("d1", "ans", "think-"));
    batcher.enqueue(makeDelta("d2", "wer", null));
    batcher.enqueue(makeDelta("d3", null, "more"));
    clock.tick();

    expect(commits).toHaveLength(1);
    expect(commits[0].content).toBe("answer");
    expect(commits[0].reasoning_content).toBe("think-more");
  });

  it("flush() commits synchronously and cancels the scheduled frame", () => {
    const commits: StreamingDeltaEvent[] = [];
    const clock = manualScheduler();
    const batcher = createStreamingDeltaBatcher(
      (delta) => commits.push(delta),
      clock.scheduler,
    );

    batcher.enqueue(makeDelta("d1", "a"));
    batcher.enqueue(makeDelta("d2", "b"));
    batcher.flush();

    expect(commits).toHaveLength(1);
    expect(commits[0].content).toBe("ab");
    // The pending frame was cancelled, so ticking must not double-commit.
    expect(clock.pendingFrames()).toBe(0);
    clock.tick();
    expect(commits).toHaveLength(1);
  });

  it("flush() is a no-op when nothing is buffered", () => {
    const commits: StreamingDeltaEvent[] = [];
    const clock = manualScheduler();
    const batcher = createStreamingDeltaBatcher(
      (delta) => commits.push(delta),
      clock.scheduler,
    );

    batcher.flush();
    expect(commits).toHaveLength(0);
  });

  it("reset() drops buffered deltas without committing", () => {
    const commits: StreamingDeltaEvent[] = [];
    const clock = manualScheduler();
    const batcher = createStreamingDeltaBatcher(
      (delta) => commits.push(delta),
      clock.scheduler,
    );

    batcher.enqueue(makeDelta("d1", "lost"));
    batcher.reset();
    clock.tick();

    expect(commits).toHaveLength(0);
    expect(clock.pendingFrames()).toBe(0);
  });

  it("preserves text byte-for-byte and order across thousands of 1-char deltas faster than 60Hz", () => {
    const commits: StreamingDeltaEvent[] = [];
    const clock = manualScheduler();
    const batcher = createStreamingDeltaBatcher(
      (delta) => commits.push(delta),
      clock.scheduler,
    );

    const total = 5000;
    let expected = "";
    for (let i = 0; i < total; i += 1) {
      const char = String.fromCharCode(97 + (i % 26));
      expected += char;
      batcher.enqueue(makeDelta(`d${i}`, char));
      // A frame only every 100 deltas => deltas arrive far faster than frames.
      if (i % 100 === 99) {
        clock.tick();
      }
    }
    batcher.flush(); // boundary flush, as a non-delta event would trigger

    // Commits are bounded by frames, not by provider chunk count.
    expect(commits.length).toBeLessThan(total);
    expect(commits.length).toBeLessThanOrEqual(total / 100 + 1);
    // Concatenating the per-frame batches reproduces the stream exactly (the
    // store folds these into one accumulating event by position).
    expect(commits.map((delta) => delta.content).join("")).toBe(expected);
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

  it("coalesces deltas across frames, then reconciles into one bubble when the final message arrives", () => {
    useEventStore.getState().clearEvents();
    const clock = manualScheduler();
    // Commit into the real store exactly as ConversationWebSocketProvider does.
    const batcher = createStreamingDeltaBatcher(
      (delta) => useEventStore.getState().addEvent(delta),
      clock.scheduler,
    );

    useEventStore.getState().addEvent(userMessage);

    // Stream one char per delta, flushing a frame only every 5 chars, so deltas
    // arrive faster than frames — the case where the UI used to fall behind.
    const streamed = "I'll start working on that.";
    [...streamed].forEach((char, i) => {
      batcher.enqueue(makeDelta(`d${i}`, char));
      if (i % 5 === 4) {
        clock.tick();
      }
    });

    // A non-delta event (the final agent message) arrives. The provider flushes
    // buffered deltas first, so the durable message can never overtake its own
    // streamed text.
    batcher.flush();
    const finalMessage: MessageEvent = {
      id: "agent-1",
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
    // The user message plus a single reconciled agent bubble — the canonical
    // final message supersedes the streamed deltas rather than duplicating them.
    expect(state.uiEvents).toHaveLength(2);
    const bubble = state.uiEvents[1] as MessageEvent;
    expect(bubble.id).toBe("agent-1");
    expect(bubble.llm_message.content).toEqual([
      { type: "text", text: "I'll start working on that. Done." },
    ]);
    // No provisional delta survives, so the streamed text renders exactly once.
    expect(state.uiEvents.some((event) => isStreamingDeltaEvent(event))).toBe(
      false,
    );
    // eventIds tracks only the two durable events, never the 27 deltas.
    expect(state.eventIds.size).toBe(2);
  });
});
