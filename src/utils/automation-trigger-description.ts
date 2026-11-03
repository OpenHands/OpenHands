import type { TFunction } from "i18next";
import { I18nKey } from "#/i18n/declaration";
import type { AutomationTrigger } from "#/types/automation";
import { formatTriggerSourceLabel } from "#/components/features/home/featured-automations/automation-run-health";
import { formatEventOn, parseCronSchedule } from "#/utils/automation-schedule";

const EVERY_N_MINUTES = /^\*\/(\d+)$/;

/** A fixed Sunday, so `weekday + day` lands on that day of the week. */
const SUNDAY_UTC = Date.UTC(2023, 0, 1);

function formatClockTime(hour: number, minute: number, locale: string) {
  return new Date(Date.UTC(1970, 0, 1, hour, minute)).toLocaleTimeString(
    locale,
    { hour: "numeric", minute: "2-digit", timeZone: "UTC" },
  );
}

function formatWeekdayName(weekday: number, locale: string) {
  return new Date(SUNDAY_UTC + weekday * 86_400_000).toLocaleDateString(
    locale,
    { weekday: "long", timeZone: "UTC" },
  );
}

/**
 * Cron shapes the service accepts but `parseCronSchedule` leaves as custom:
 * every minute, every N minutes, and once an hour.
 */
function describeSubDailyCron(cron: string, t: TFunction): string | null {
  const fields = cron.trim().split(/\s+/);
  if (fields.length !== 5) return null;
  const [minute, ...rest] = fields;
  if (rest.some((field) => field !== "*")) return null;

  if (minute === "*") return t(I18nKey.AUTOMATIONS$SCHEDULE$EVERY_MINUTE);
  const interval = minute.match(EVERY_N_MINUTES);
  if (interval) {
    const count = Number(interval[1]);
    return count === 1
      ? t(I18nKey.AUTOMATIONS$SCHEDULE$EVERY_MINUTE)
      : t(I18nKey.AUTOMATIONS$SCHEDULE$EVERY_N_MINUTES, { count });
  }
  if (/^\d+$/.test(minute) && Number(minute) < 60) {
    return t(I18nKey.AUTOMATIONS$SCHEDULE$HOURLY_AT, {
      minute: minute.padStart(2, "0"),
    });
  }
  return null;
}

function describeCron(
  cron: string,
  t: TFunction,
  locale: string,
): string | null {
  const parsed = parseCronSchedule(cron);
  if (parsed.kind === "custom") return describeSubDailyCron(cron, t);

  const time = formatClockTime(parsed.hour, parsed.minute, locale);
  if (parsed.kind === "daily") {
    return t(I18nKey.AUTOMATIONS$SCHEDULE$DAILY_AT, { time });
  }
  if (parsed.kind === "weekdays") {
    return t(I18nKey.AUTOMATIONS$SCHEDULE$WEEKDAYS_AT, { time });
  }
  return t(I18nKey.AUTOMATIONS$SCHEDULE$WEEKLY_AT, {
    time,
    weekday: formatWeekdayName(parsed.weekday ?? 0, locale),
  });
}

/**
 * The one way every automation surface words a schedule trigger: a localized
 * sentence when the cron has a common shape, then the service's own
 * description, then the raw expression. Null for event triggers.
 */
export function describeSchedule(
  trigger: AutomationTrigger,
  t: TFunction,
  locale: string,
): string | null {
  if (trigger.type === "event") return null;
  const cron = trigger.schedule?.trim() ?? "";
  return (
    (cron ? describeCron(cron, t, locale) : null) ||
    trigger.schedule_human?.trim() ||
    cron ||
    null
  );
}

/**
 * One-line summary of what starts an automation, shared by the detail page
 * and the list card: "Every day at 9:00 AM" or "GitHub · release.published".
 */
export function describeTrigger(
  trigger: AutomationTrigger,
  t: TFunction,
  locale: string,
): string {
  if (trigger.type !== "event") {
    return describeSchedule(trigger, t, locale) ?? trigger.type;
  }
  const source = trigger.source?.trim();
  return [
    source ? formatTriggerSourceLabel(source) : null,
    trigger.on ? formatEventOn(trigger.on) : null,
  ]
    .filter(Boolean)
    .join(" · ");
}
