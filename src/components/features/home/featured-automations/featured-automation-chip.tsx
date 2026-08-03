import { useTranslation } from "react-i18next";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import type { LatestAutomationRunState } from "#/hooks/query/use-latest-automation-runs";
import { I18nKey } from "#/i18n/declaration";
import ClockIcon from "#/icons/clock.svg?react";
import GlobeIcon from "#/icons/globe.svg?react";
import { AutomationRunStatus, type Automation } from "#/types/automation";
import { formatRelativeTime } from "#/utils/format-relative-time";
import { cn } from "#/utils/utils";
import { AutomationHealthIndicator } from "./automation-health-indicator";
import {
  deriveRunHealth,
  getLastRunTimestamp,
  getRunHealthLabelKey,
  getTriggerSummary,
} from "./automation-run-health";

interface FeaturedAutomationChipProps {
  automation: Automation;
  runState: LatestAutomationRunState;
  isFeatured: boolean;
  onToggle: (automationId: string) => void;
}

function getStatusLabelKey(runState: LatestAutomationRunState): I18nKey {
  if (runState.isError) return I18nKey.FEATURED_AUTOMATIONS$STATUS_UNAVAILABLE;
  return getRunHealthLabelKey(deriveRunHealth(runState));
}

function ChipTooltip({
  automation,
  runState,
}: {
  automation: Automation;
  runState: LatestAutomationRunState;
}) {
  const { t, i18n } = useTranslation("openhands");
  const { latestRun } = runState;
  const timestamp = latestRun ? getLastRunTimestamp(latestRun) : null;
  const TriggerIcon =
    automation.trigger.type === "event" ? GlobeIcon : ClockIcon;

  return (
    <div className="max-w-64 space-y-1.5 text-left">
      <p className="font-semibold">{automation.name}</p>
      <p className="flex items-center gap-1.5">
        <TriggerIcon className="size-3 shrink-0" aria-hidden="true" />
        {getTriggerSummary(automation)}
      </p>
      <div className="flex items-center justify-between gap-3">
        <span className="flex items-center gap-1.5">
          <AutomationHealthIndicator health={deriveRunHealth(runState)} />
          {t(getStatusLabelKey(runState))}
        </span>
        {timestamp ? (
          <span>{formatRelativeTime(timestamp, i18n.language, t)}</span>
        ) : null}
      </div>
      {latestRun?.status === AutomationRunStatus.FAILED &&
      latestRun.error_detail ? (
        <p className="line-clamp-2 text-[var(--oh-status-error)]">
          {latestRun.error_detail}
        </p>
      ) : null}
    </div>
  );
}

export function FeaturedAutomationChip({
  automation,
  runState,
  isFeatured,
  onToggle,
}: FeaturedAutomationChipProps) {
  const { t } = useTranslation("openhands");

  return (
    <StyledTooltip
      content={<ChipTooltip automation={automation} runState={runState} />}
      placement="bottom"
    >
      <button
        type="button"
        aria-pressed={isFeatured}
        onClick={() => onToggle(automation.id)}
        className={cn(
          "inline-flex items-center gap-2 rounded-md border px-3 py-2 text-sm text-[var(--oh-foreground)] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--oh-focus)]",
          isFeatured
            ? "border-[var(--oh-focus)] bg-[var(--oh-interactive-selected)] shadow-inner"
            : "border-[var(--oh-border)] bg-[var(--oh-surface-raised)] hover:bg-[var(--oh-interactive-hover)]",
        )}
      >
        <AutomationHealthIndicator health={deriveRunHealth(runState)} />
        <span className="max-w-[12rem] truncate">{automation.name}</span>
        <span className="sr-only">{t(getStatusLabelKey(runState))}</span>
      </button>
    </StyledTooltip>
  );
}
