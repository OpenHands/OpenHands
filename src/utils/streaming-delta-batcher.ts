import { StreamingDeltaEvent } from "#/types/agent-server/core/events/streaming-delta-event";
import { mergeStreamingDeltaEvent } from "#/utils/handle-event-for-ui";

/** Schedules a single deferred callback (defaults to the animation frame). */
export interface DeltaFlushScheduler {
  schedule: (callback: () => void) => number;
  cancel: (handle: number) => void;
}

const defaultScheduler: DeltaFlushScheduler =
  typeof requestAnimationFrame === "function"
    ? {
        schedule: (callback) => requestAnimationFrame(callback),
        cancel: (handle) => cancelAnimationFrame(handle),
      }
    : {
        schedule: (callback) => setTimeout(callback, 16) as unknown as number,
        cancel: (handle) => clearTimeout(handle),
      };

export interface StreamingDeltaBatcher {
  /** Buffer a delta; a flush is scheduled for the next frame if not already. */
  enqueue: (event: StreamingDeltaEvent) => void;
  /** Commit buffered deltas now. Call before any non-delta event. */
  flush: () => void;
  /** Drop buffered deltas without committing. Call on unmount / conversation switch. */
  reset: () => void;
}

/**
 * Coalesces adjacent `StreamingDeltaEvent`s and commits them at most once per
 * animation frame, so a fast model can't force a store commit + re-render per
 * token. Callers MUST `flush()` before any non-delta event so a
 * durable message/action can't render ahead of its own streamed text.
 */
export function createStreamingDeltaBatcher(
  commit: (event: StreamingDeltaEvent) => void,
  scheduler: DeltaFlushScheduler = defaultScheduler,
): StreamingDeltaBatcher {
  let pending: StreamingDeltaEvent[] = [];
  let frame: number | null = null;

  const cancelFrame = () => {
    if (frame !== null) {
      scheduler.cancel(frame);
      frame = null;
    }
  };

  const flush = () => {
    cancelFrame();
    if (pending.length === 0) {
      return;
    }
    const batch = pending;
    pending = [];
    commit(
      batch.reduce((merged, delta) => mergeStreamingDeltaEvent(delta, merged)),
    );
  };

  return {
    enqueue: (event) => {
      pending.push(event);
      if (frame === null) {
        frame = scheduler.schedule(flush);
      }
    },
    flush,
    reset: () => {
      cancelFrame();
      pending = [];
    },
  };
}
