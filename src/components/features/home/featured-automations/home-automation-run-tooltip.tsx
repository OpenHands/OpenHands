import type { ReactNode } from "react";
import { Zap } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { LatestAutomationRunState } from "#/hooks/query/use-latest-automation-runs";
import { I18nKey } from "#/i18n/declaration";
import ClockIcon from "#/icons/clock.svg?react";
import { AutomationRunStatus, type Automation } from "#/types/automation";
import { formatRelativeTime } from "#/utils/format-relative-time";
import { AutomationHealthIndicator } from "./automation-health-indicator";
import {
  deriveRunHealth,
  getLastRunTimestamp,
  getRunHealthLabelKey,
  getTriggerSummary,
} from "./automation-run-health";

export function getRunStatusLabelKey(
  runState: LatestAutomationRunState,
): I18nKey {
  if (runState.isError) return I18nKey.FEATURED_AUTOMATIONS$STATUS_UNAVAILABLE;
  return getRunHealthLabelKey(deriveRunHealth(runState));
}

function PreviewRow({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="flex items-start gap-2 text-xs">
      <span className="w-20 shrink-0 text-[var(--oh-muted)]">{label}</span>
      <span className="min-w-0 flex-1 break-words text-[var(--oh-foreground)]">
        {children}
      </span>
    </div>
  );
}

/**
 * Left-nav-style hovercard body for an automation's latest-run health.
 * Matches `ConversationCardPreview` layout: title + label/value rows.
 */
export function HomeAutomationRunTooltip({
  automation,
  runState,
}: {
  automation: Automation;
  runState: LatestAutomationRunState;
}) {
  const { t, i18n } = useTranslation("openhands");
  const { latestRun } = runState;
  const timestamp = latestRun ? getLastRunTimestamp(latestRun) : null;
  const TriggerIcon = automation.trigger.type === "event" ? Zap : ClockIcon;
  const health = deriveRunHealth(runState);

  return (
    <div className="flex w-[280px] flex-col gap-3 p-3">
      <span className="break-words text-sm font-medium text-white">
        {automation.name}
      </span>

      <dl className="flex flex-col gap-1.5">
        <PreviewRow label={t(I18nKey.AUTOMATIONS$DETAIL$TRIGGER)}>
          <span className="inline-flex min-w-0 items-start gap-1.5">
            <TriggerIcon
              className="mt-0.5 size-3 shrink-0"
              aria-hidden="true"
            />
            <span className="break-all">{getTriggerSummary(automation)}</span>
          </span>
        </PreviewRow>

        <PreviewRow label={t(I18nKey.COMMON$STATUS)}>
          <span className="inline-flex min-w-0 items-center gap-1.5">
            <AutomationHealthIndicator health={health} />
            <span>{t(getRunStatusLabelKey(runState))}</span>
          </span>
        </PreviewRow>

        {timestamp ? (
          <PreviewRow label={t(I18nKey.AUTOMATIONS$DETAIL$LAST_RUN)}>
            {formatRelativeTime(timestamp, i18n.language, t)}
          </PreviewRow>
        ) : null}

        {latestRun?.status === AutomationRunStatus.FAILED &&
        latestRun.error_detail ? (
          <PreviewRow label={t(I18nKey.STATUS$ERROR)}>
            <span className="line-clamp-3 text-[var(--oh-status-error)]">
              {latestRun.error_detail}
            </span>
          </PreviewRow>
        ) : null}
      </dl>
    </div>
  );
}
