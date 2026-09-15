import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";
import { getDisablementReasonDisplay } from "#/utils/automation-disabled-reason";
import {
  formatRelativeTime,
  isInvalidTimestamp,
} from "#/utils/format-relative-time";
import PauseIcon from "#/icons/pause.svg?react";

interface DisabledReasonBannerProps {
  automation: Automation;
}

/**
 * Surfaces the reason an inactive automation was last disabled.
 *
 * The automation service records `disabled_reason`/`disabled_detail`/
 * `disabled_at` on every disable (manual toggle, manual delete, or automatic
 * pause for failing runs) and clears them on every re-enable, so when the
 * automation is inactive and carries a reason, that reason is the latest
 * reason it went inactive. Shown on the automation detail page below the
 * header so the user knows *why* an automation is paused, not just that it is.
 */
export function DisabledReasonBanner({
  automation,
}: DisabledReasonBannerProps) {
  const { t, i18n } = useTranslation("openhands");

  const display = getDisablementReasonDisplay(automation, t);
  if (!display) return null;

  const showTimestamp = !isInvalidTimestamp(automation.disabled_at);

  return (
    <div
      data-testid="automation-disabled-reason-banner"
      className="flex flex-col gap-2 rounded-2xl border border-[var(--oh-border)] bg-[var(--oh-surface)] px-5 py-4"
    >
      <div className="flex items-center gap-2">
        <PauseIcon className="size-4 shrink-0 text-muted" aria-hidden />
        <h3 className="text-sm font-medium text-content">
          {t(I18nKey.AUTOMATIONS$DETAIL$DISABLED_REASON_HEADING)}
        </h3>
      </div>
      <p
        data-testid="automation-disabled-reason-text"
        className="text-sm leading-5 text-[var(--oh-text-secondary)]"
      >
        {display.text}
      </p>
      {showTimestamp ? (
        <p
          data-testid="automation-disabled-reason-timestamp"
          className="text-xs text-muted"
        >
          {t(I18nKey.AUTOMATIONS$DETAIL$DISABLED_AT, {
            time: formatRelativeTime(
              automation.disabled_at as string,
              i18n.language,
              t,
            ),
          })}
        </p>
      ) : null}
    </div>
  );
}
