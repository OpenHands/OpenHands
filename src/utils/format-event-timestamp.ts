export function formatEventTimestamp(
  timestamp?: string,
  locale?: string,
): string | null {
  if (!timestamp) return null;

  // Agent-server events carry datetime.now().isoformat(), which has no offset.
  // Left alone the browser reads that wall-clock as its own local time, so an
  // event recorded in a UTC sandbox renders hours off for anyone outside UTC.
  const utc = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(timestamp)
    ? timestamp
    : `${timestamp}Z`;
  const date = new Date(utc);
  if (Number.isNaN(date.getTime())) return null;

  return date.toLocaleString(locale, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}
