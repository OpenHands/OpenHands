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
   * Identity the automation runs as: a person's email (or raw user id when
   * the email could not be resolved), or a service account's name (its raw id
   * when a manager's lookup failed, or a generic label for a member).
   * Omitted/null hides the field (local backends, or while the lookup is
   * still in flight).
   */
  runsAs?: string | null;
  /** `runsAs` is a service account: show the SA badge next to it. */
  runsAsServiceAccount?: boolean;
  /**
   * The person who created the automation, resolved like `runsAs`. Passed
   * only when it differs from `runsAs` (an automation that runs as a service
   * account); omitted/null hides the field.
   */
  createdBy?: string | null;
}

/** How the automation runs, rather than what it does or when. */
export function SettingsSection({
  automation,
  runsAs,
  runsAsServiceAccount = false,
  createdBy,
}: SettingsSectionProps) {
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
            <span className="flex flex-wrap items-center gap-2">
              <span className="break-all">{runsAs}</span>
              {runsAsServiceAccount ? (
                <span
                  data-testid="service-account-badge"
                  title={t(I18nKey.AUTOMATIONS$SERVICE_ACCOUNT_LABEL)}
                  className="rounded-md bg-surface-raised px-1.5 py-0.5 text-xs font-medium text-muted"
                >
                  {t(I18nKey.AUTOMATIONS$SERVICE_ACCOUNT_BADGE)}
                </span>
              ) : null}
            </span>
          </ConfigField>
        ) : null}

        {createdBy ? (
          <ConfigField
            icon={<UserIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$CREATED_BY)}
          >
            <span className="break-all">{createdBy}</span>
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
