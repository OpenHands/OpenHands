/**
 * Resume cursor for `/sockets/session/{id}`.
 *
 * Durable frames are not delivered in `seq` order — a state update can overtake
 * the message persisted just before it — so resuming from the highest seq seen
 * would skip whatever was still in flight when the socket dropped. The cursor
 * is instead the highest seq with nothing missing below it.
 */
export interface SeqCursor {
  /** Highest seq with no gap below it, or `null` before the first connect. */
  readonly value: number | null;
  /** Begin a connection that replays everything after `base`. */
  start: (base: number) => void;
  observe: (seq: number) => void;
  clear: () => void;
}

/**
 * How far past a gap to wait before stepping over it. The server skips events
 * it cannot read, and that seq never arrives — on this connection or any
 * later one — so an unbounded wait would freeze the cursor at the hole and
 * replay from it on every reconnect.
 */
const MAX_AHEAD = 1024;

export function createSeqCursor(
  /**
   * Reports the `seq` interval stepped over when a hole stays open past
   * `MAX_AHEAD`. The skipped events are unreadable on the server, so they
   * cannot be refetched — the callback exists so the skip is recorded
   * instead of silent.
   */
  onGap?: (from: number, to: number) => void,
): SeqCursor {
  let value: number | null = null;
  let ahead = new Set<number>();

  const drain = () => {
    while (value !== null && ahead.has(value + 1)) {
      value += 1;
      ahead.delete(value);
    }
  };

  return {
    get value() {
      return value;
    },
    start: (base) => {
      value = base;
      // Anything seen past a gap is replayed on this connection and deduped.
      ahead = new Set();
    },
    observe: (seq) => {
      if (value === null || seq <= value) {
        return;
      }
      ahead.add(seq);
      drain();
      if (ahead.size > MAX_AHEAD) {
        // Step over the hole: the skipped event is unreadable on replay too.
        // Record the interval so the skip is explicit.
        const skippedFrom = value + 1;
        const skippedTo = Math.min(...ahead) - 1;
        value = skippedTo;
        drain();
        onGap?.(skippedFrom, skippedTo);
      }
    },
    clear: () => {
      value = null;
      ahead = new Set();
    },
  };
}
