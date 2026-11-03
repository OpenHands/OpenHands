import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";
import CogIcon from "#/icons/cog.svg?react";
import UserIcon from "#/icons/user.svg?react";
import BellIcon from "#/icons/bell.svg?react";
import ClockIcon from "#/icons/clock.svg?react";
import CalendarIcon from "#/icons/calendar.svg?react";
import { formatCompactDuration } from "#/manifests/automation-insights";
import { formatDate } from "#/utils/format-relative-time";
import { SectionCard } from "./section-card";
import { ConfigField } from "./config-field";

interface SettingsSectionProps {
  automation: Automation;
  /**
   * Identity the automation runs as — the creator's email, or their raw user
   * id when the email could not be resolved. Omitted/null hides the field
   * (local backends, or while the lookup is still in flight).
   */
  runsAs?: string | null;
}

/** How the automation runs, rather than what it does or when. */
export function SettingsSection({ automation, runsAs }: SettingsSectionProps) {
  const { t, i18n } = useTranslation("openhands");

  return (
    <SectionCard
      testId="automation-settings-section"
      icon={<CogIcon className="size-4" />}
      title={t(I18nKey.AUTOMATIONS$SECTION$SETTINGS)}
    >
      <div className="grid grid-cols-2 gap-x-4 gap-y-5">
        {runsAs ? (
          <ConfigField
            icon={<UserIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$RUNS_AS)}
          >
            <span className="break-all">{runsAs}</span>
          </ConfigField>
        ) : null}

        {automation.timeout ? (
          <ConfigField
            icon={<ClockIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$TIMEOUT)}
          >
            {formatCompactDuration(automation.timeout * 1000)}
          </ConfigField>
        ) : null}

        {automation.notification ? (
          <ConfigField
            icon={<BellIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$NOTIFICATION)}
          >
            {automation.notification}
          </ConfigField>
        ) : null}

        <ConfigField
          icon={<CalendarIcon className="size-3.5" />}
          label={t(I18nKey.AUTOMATIONS$DETAIL$CREATED)}
        >
          {formatDate(automation.created_at, i18n.language)}
        </ConfigField>
      </div>
    </SectionCard>
  );
}
