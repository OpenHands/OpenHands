const MINUTE_MS = 60_000;
const HOUR_MS = 60 * MINUTE_MS;
const DAY_MS = 24 * HOUR_MS;
const WEEK_MS = 7 * DAY_MS;

function parseEventTimestamp(timestamp?: string): Date | null {
  if (!timestamp) return null;

  // Agent-server timestamps without an offset are UTC, not browser-local time.
  const utc = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(timestamp)
    ? timestamp
    : `${timestamp}Z`;
  const date = new Date(utc);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatEventTimestamp(
  timestamp?: string,
  locale?: string,
): string | null {
  return (
    parseEventTimestamp(timestamp)?.toLocaleString(locale, {
      dateStyle: "medium",
      timeStyle: "short",
    }) ?? null
  );
}

export function formatRelativeEventTimestamp(
  timestamp: string | undefined,
  locale: string | undefined,
  justNowLabel: string,
  now = Date.now(),
): string | null {
  const date = parseEventTimestamp(timestamp);
  if (!date) return null;
  const elapsed = Math.max(0, now - date.getTime());
  if (elapsed < MINUTE_MS) return justNowLabel;
  if (elapsed >= WEEK_MS) {
    return date.toLocaleDateString(locale, {
      month: "short",
      day: "numeric",
      ...(date.getFullYear() !== new Date(now).getFullYear()
        ? { year: "numeric" }
        : {}),
    });
  }

  const unit = elapsed < HOUR_MS ? "minute" : elapsed < DAY_MS ? "hour" : "day";
  const divisor =
    unit === "minute" ? MINUTE_MS : unit === "hour" ? HOUR_MS : DAY_MS;
  return new Intl.RelativeTimeFormat(locale, {
    numeric: "always",
    style: "long",
  }).format(-Math.floor(elapsed / divisor), unit);
}
