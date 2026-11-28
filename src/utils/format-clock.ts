/**
 * Format a number of whole seconds as a minutes:seconds clock, e.g. "00:12",
 * "01:05", "61:01". Minutes are not wrapped at 60 so long replies keep counting
 * up rather than rolling back to zero.
 */
export function formatClock(totalSeconds: number): string {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}
