import { useEffect, useRef } from "react";
import useMetricsStore from "#/stores/metrics-store";
import { useEventStore } from "#/stores/use-event-store";
import { isCondensationEvent } from "#/types/agent-server/type-guards";

/** How long to wait after a Condensation event for a lower per_turn_token. */
const METRICS_SETTLE_MS = 2_500;
/** Give up waiting for condensation / metrics and surface whatever we have. */
const DEFAULT_TIMEOUT_MS = 90_000;

export type ContextCompactionOutcome = "compacted" | "no_change" | "timeout";

export interface ContextCompactionResult {
  beforeToken: number;
  afterToken: number;
  savedToken: number;
  /**
   * "compacted" — a Condensation event landed and per_turn_token dropped.
   * "no_change" — a Condensation event landed but no token drop was measured.
   * "timeout" — no Condensation event arrived in time; compaction did not
   * happen (or was never processed), so this is a failure, not a no-op.
   */
  outcome: ContextCompactionOutcome;
}

export function buildContextCompactionResult(
  beforeToken: number,
  afterToken: number,
  outcome: ContextCompactionOutcome,
): ContextCompactionResult {
  return {
    beforeToken,
    afterToken,
    savedToken: Math.max(0, beforeToken - afterToken),
    outcome,
  };
}

interface UseAwaitContextCompactionOptions {
  /**
   * Snapshot of `per_turn_token` taken when compaction was requested.
   * Pass `null` when not awaiting a result.
   */
  beforeToken: number | null;
  /**
   * Event ids that predate the compaction *request*. The condense POST can
   * return only after the server already emitted its Condensation event, so
   * snapshotting "known" ids when this effect starts would miss fast
   * condensations; the baseline must be captured before the request fires.
   * When omitted, falls back to the ids present when the effect starts.
   */
  baselineEventIds?: Set<string | number> | null;
  onComplete: (result: ContextCompactionResult) => void;
  timeoutMs?: number;
}

/**
 * Waits for a post-request Condensation event (and ideally a lower
 * `per_turn_token` in the live metrics store), then reports how many tokens
 * were freed. The HTTP `/condense` ack only means work *started*.
 */
export function useAwaitContextCompaction({
  beforeToken,
  baselineEventIds,
  onComplete,
  timeoutMs = DEFAULT_TIMEOUT_MS,
}: UseAwaitContextCompactionOptions) {
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  useEffect(() => {
    if (beforeToken === null) {
      return undefined;
    }

    let completed = false;
    let sawCondensation = false;
    let settleTimer: ReturnType<typeof setTimeout> | undefined;
    const knownEventIds = baselineEventIds
      ? new Set(baselineEventIds)
      : new Set(useEventStore.getState().eventIds);

    const readAfterToken = () =>
      useMetricsStore.getState().usage?.per_turn_token ?? beforeToken;

    const finish = (afterToken: number, outcome: ContextCompactionOutcome) => {
      if (completed) {
        return;
      }
      completed = true;
      onCompleteRef.current(
        buildContextCompactionResult(beforeToken, afterToken, outcome),
      );
    };

    const finishFromMetricsIfReady = () => {
      const afterToken = readAfterToken();
      if (sawCondensation && afterToken < beforeToken) {
        finish(afterToken, "compacted");
        return true;
      }
      return false;
    };

    const scheduleSettleFinish = () => {
      clearTimeout(settleTimer);
      settleTimer = setTimeout(() => {
        finish(readAfterToken(), "no_change");
      }, METRICS_SETTLE_MS);
    };

    const scanForNewCondensation = () => {
      for (const event of useEventStore.getState().events) {
        const eventId = "id" in event ? event.id : undefined;
        if (eventId !== undefined) {
          if (knownEventIds.has(eventId)) {
            continue;
          }
          knownEventIds.add(eventId);
        }
        if (!isCondensationEvent(event)) {
          continue;
        }
        sawCondensation = true;
        if (!finishFromMetricsIfReady()) {
          scheduleSettleFinish();
        }
        return;
      }
    };

    scanForNewCondensation();

    const unsubscribeEvents = useEventStore.subscribe(() => {
      if (completed) {
        return;
      }
      scanForNewCondensation();
    });

    const unsubscribeMetrics = useMetricsStore.subscribe(() => {
      if (completed) {
        return;
      }
      if (finishFromMetricsIfReady()) {
        clearTimeout(settleTimer);
      }
    });

    const timeoutTimer = setTimeout(() => {
      // No Condensation event in time means the compaction never landed —
      // a failure, not a "nothing to compact" no-op.
      finish(readAfterToken(), sawCondensation ? "no_change" : "timeout");
    }, timeoutMs);

    return () => {
      completed = true;
      unsubscribeEvents();
      unsubscribeMetrics();
      clearTimeout(settleTimer);
      clearTimeout(timeoutTimer);
    };
  }, [beforeToken, baselineEventIds, timeoutMs]);
}
