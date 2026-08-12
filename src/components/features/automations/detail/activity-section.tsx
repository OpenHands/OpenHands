import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import ActivityIcon from "#/icons/activity.svg?react";
import CalendarIcon from "#/icons/calendar.svg?react";
import ClockIcon from "#/icons/clock.svg?react";
import { formatDate, formatRelativeTime } from "#/utils/format-relative-time";
import { SectionCard } from "./section-card";
import { ConfigField } from "./config-field";

interface ActivitySectionProps {
  createdAt: string;
  lastRunAt: string | null | undefined;
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
