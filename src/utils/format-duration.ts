/**
 * Formats an elapsed duration in milliseconds as a compact, human-readable
 * string, e.g. `~12s`, `~1m 5s`, `~3m`, `~2h 5m`.
 *
 * Sub-second durations are rounded to whole seconds. Negative or non-finite
 * inputs fall back to `~0s`.
 */
export const formatDurationMs = (ms: number): string => {
  if (!Number.isFinite(ms) || ms < 0) return "~0s";

  const totalSeconds = Math.round(ms / 1000);
  if (totalSeconds < 60) return `~${totalSeconds}s`;

  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes < 60) {
    return seconds === 0 ? `~${minutes}m` : `~${minutes}m ${seconds}s`;
  }

  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return remainingMinutes === 0
    ? `~${hours}h`
    : `~${hours}h ${remainingMinutes}m`;
};
