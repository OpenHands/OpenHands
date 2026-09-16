import type { DeltaFrame } from "#/types/agent-server/session-frames";

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
  /** Buffer a delta frame; a flush is scheduled for the next frame if not already. */
  enqueue: (frame: DeltaFrame) => void;
  /** Commit buffered deltas now. Call before any other frame. */
  flush: () => void;
  /** Drop buffered deltas without committing. Call on unmount / conversation switch. */
  reset: () => void;
}

/**
 * Commits buffered `delta` frames at most once per animation frame, so a fast
 * model can't force a store commit + re-render per token. Callers MUST
 * `flush()` before any other frame so an `item_started`, an abort or a durable
 * message can't overtake text that was streamed ahead of it.
 */
export function createStreamingDeltaBatcher(
  commit: (frames: DeltaFrame[]) => void,
  scheduler: DeltaFlushScheduler = defaultScheduler,
): StreamingDeltaBatcher {
  let pending: DeltaFrame[] = [];
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
    commit(batch);
  };

  return {
    enqueue: (incoming) => {
      pending.push(incoming);
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
