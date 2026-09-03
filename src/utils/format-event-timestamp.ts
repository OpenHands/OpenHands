export function formatEventTimestamp(
  timestamp?: string,
  locale?: string,
): string | null {
  if (!timestamp) return null;

  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return null;

  return date.toLocaleString(locale, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}
