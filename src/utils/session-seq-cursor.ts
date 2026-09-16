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

export function createSeqCursor(): SeqCursor {
  let value: number | null = null;
  let ahead = new Set<number>();

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
      while (ahead.has(value + 1)) {
        value += 1;
        ahead.delete(value);
      }
    },
    clear: () => {
      value = null;
      ahead = new Set();
    },
  };
}
