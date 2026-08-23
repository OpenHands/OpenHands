import React from "react";
import { useSecondsTick } from "#/hooks/use-seconds-tick";
import { formatTimeDelta } from "#/utils/format-time-delta";

interface EventElapsedTimeProps {
  /** ISO timestamp when the tool call started (ActionEvent.timestamp). */
  startTimestamp: string;
  /**
   * ISO timestamp when the tool call completed (ObservationEvent.timestamp).
   * When absent the component shows a live-updating elapsed counter.
   * When present the component shows a static final duration.
   */
  endTimestamp?: string;
}

/**
 * Compact duration formatter for a pre-computed millisecond value.
 * Mirrors the output format of `formatTimeDelta` (0s, 1s, 2m, 1h, …) but
 * accepts a duration directly so we avoid a round-trip through a fake Date.
 */
function formatDurationMs(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  const months = Math.floor(days / 30);
  const years = Math.floor(months / 12);

  if (seconds < 60) return `${seconds}s`;
  if (minutes < 60) return `${minutes}m`;
  if (hours < 24) return `${hours}h`;
  if (days < 30) return `${days}d`;
  if (months < 12) return `${months}mo`;
  return `${years}y`;
}

/**
 * Displays the execution duration for a single tool-call event card.
 *
 * - While an ActionEvent is in flight (no `endTimestamp`): shows a live
 *   counter that updates every second via `useSecondsTick`.
 * - Once the ObservationEvent arrives (`endTimestamp` present): shows the
 *   final static duration computed from the two event timestamps.
 *
 * The duration is computed from event timestamps only, so it reflects
 * server-side elapsed time rather than client wall-clock drift. Negative
 * deltas (possible with minor clock skew) are clamped to zero.
 *
 * Returns null for invalid / unparseable timestamps so the card degrades
 * gracefully without throwing.
 */
export function EventElapsedTime({
  startTimestamp,
  endTimestamp,
}: EventElapsedTimeProps) {
  const isRunning = !endTimestamp;

  // nowMs is set inside useState/useEffect — never called during render itself.
  const nowMs = useSecondsTick(isRunning);

  const startMs = new Date(startTimestamp).getTime();
  if (Number.isNaN(startMs)) {
    return null;
  }

  let label: string;

  if (endTimestamp) {
    const endMs = new Date(endTimestamp).getTime();
    if (Number.isNaN(endMs)) {
      return null;
    }
    // Clamp to guard against minor server clock skew producing a negative delta.
    label = formatDurationMs(Math.max(0, endMs - startMs));
  } else {
    // Guard against server clock skew: if the action's timestamp is ahead of
    // the client clock, show "0s" rather than a negative duration.
    label = nowMs >= startMs ? formatTimeDelta(startTimestamp) : "0s";
  }

  return (
    <time
      data-testid="event-elapsed-time"
      className="text-xs text-[var(--oh-muted)] ml-2 tabular-nums flex-shrink-0"
    >
      {label}
    </time>
  );
}
