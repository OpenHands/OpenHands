import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import ActivityIcon from "#/icons/activity.svg?react";
import CalendarIcon from "#/icons/calendar.svg?react";
import ClockIcon from "#/icons/clock.svg?react";
import { SectionCard } from "./section-card";
import { ConfigField } from "./config-field";

interface ActivitySectionProps {
  createdAt: string;
  lastRunAt: string | null | undefined;
}

function formatDate(dateStr: string, locale: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString(locale, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function formatRelativeTime(
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

export function ActivitySection({
  createdAt,
  lastRunAt,
}: ActivitySectionProps) {
  const { t, i18n } = useTranslation("openhands");
  const locale = i18n.language;

  return (
    <SectionCard
      icon={<ActivityIcon className="size-4" />}
      title={t(I18nKey.AUTOMATIONS$DETAIL$ACTIVITY)}
    >
      <div className="grid grid-cols-2 gap-x-4">
        <ConfigField
          icon={<CalendarIcon className="size-3.5" />}
          label={t(I18nKey.AUTOMATIONS$DETAIL$CREATED)}
        >
          {formatDate(createdAt, locale)}
        </ConfigField>

        <ConfigField
          icon={<ClockIcon className="size-3.5" />}
          label={t(I18nKey.AUTOMATIONS$DETAIL$LAST_RUN)}
        >
          {lastRunAt
            ? formatRelativeTime(lastRunAt, locale, t)
            : t(I18nKey.AUTOMATIONS$DETAIL$TIME_NEVER)}
        </ConfigField>
      </div>
    </SectionCard>
  );
}
