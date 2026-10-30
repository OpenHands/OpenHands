/**
 * Epoch milliseconds for an event timestamp, or `NaN` when it is unparseable.
 *
 * Agent-server events carry datetime.now().isoformat(), which has no offset.
 * Left alone the browser reads that wall-clock as its own local time, so an
 * event recorded in a UTC sandbox lands hours off for anyone outside UTC.
 * Timestamps that already carry `Z` or an offset are read as-is.
 */
export function parseEventTimestamp(timestamp: string): number {
  const utc = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(timestamp)
    ? timestamp
    : `${timestamp}Z`;
  return Date.parse(utc);
}

export function formatEventTimestamp(
  timestamp?: string,
  locale?: string,
): string | null {
  if (!timestamp) return null;

  const time = parseEventTimestamp(timestamp);
  if (Number.isNaN(time)) return null;

  return new Date(time).toLocaleString(locale, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}
