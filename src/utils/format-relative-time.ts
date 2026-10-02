import { I18nKey } from "#/i18n/declaration";

export function formatDate(dateStr: string, locale: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString(locale, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/**
 * Whether a timestamp is one nothing can be computed from. Besides an
 * unparseable string, that includes the epoch: the backend leaves unset
 * datetimes at zero, and formatting one produces "Jan 1, 1970" rather than
 * an honest "unknown".
 */
export function isInvalidTimestamp(
  dateStr: string | null | undefined,
): boolean {
  if (!dateStr) return true;
  const parsed = new Date(dateStr).getTime();
  return Number.isNaN(parsed) || parsed === 0;
}

export function formatRelativeTime(
  dateStr: string,
  locale: string,
  t: (key: I18nKey, options?: Record<string, unknown>) => string,
): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const diffMs = now - then;
  const diffMins = Math.floor(diffMs / 60_000);
  const diffHours = Math.floor(diffMs / 3_600_000);
  const diffDays = Math.floor(diffMs / 86_400_000);

  if (diffMins < 1) return t(I18nKey.AUTOMATIONS$DETAIL$TIME_JUST_NOW);
  if (diffMins < 60)
    return t(I18nKey.AUTOMATIONS$DETAIL$TIME_MINUTES_AGO, { count: diffMins });
  if (diffHours < 24)
    return t(I18nKey.AUTOMATIONS$DETAIL$TIME_HOURS_AGO, { count: diffHours });
  if (diffDays === 1) return t(I18nKey.AUTOMATIONS$DETAIL$TIME_YESTERDAY);
  if (diffDays < 7)
    return t(I18nKey.AUTOMATIONS$DETAIL$TIME_DAYS_AGO, { count: diffDays });
  return formatDate(dateStr, locale);
}
