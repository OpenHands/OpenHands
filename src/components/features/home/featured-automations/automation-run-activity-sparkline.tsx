import { Tooltip } from "@heroui/react";
import { useEffect, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { NavigationLink } from "#/components/shared/navigation-link";
import { AUTOMATION_RUN_ACTIVITY_LIMIT } from "#/hooks/query/use-latest-automation-runs";
import { isInFlightAutomationRun } from "#/hooks/use-home-automation-actions";
import { I18nKey } from "#/i18n/declaration";
import { AutomationRunStatus, type AutomationRun } from "#/types/automation";
import { formatRelativeTime } from "#/utils/format-relative-time";
import { cn } from "#/utils/utils";
import { getLastRunTimestamp } from "./automation-run-health";
import {
  AUTOMATION_RUN_ACTIVITY_TRACK_PX,
  barColorClassForStatus,
  durationMsToSparklineBarHeightPx,
  formatDurationForTitle,
  getAutomationRunDurationMs,
  getAutomationRunStatusLabelKey,
} from "./automation-run-activity-metrics";

interface AutomationRunActivitySparklineProps {
  automationId: string;
  /** Newest-first run list (same order as the runs API). */
  runs: readonly AutomationRun[];
  testId?: string;
}

function TooltipRow({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="flex items-start gap-2 text-xs">
      <span className="w-16 shrink-0 text-[var(--oh-muted)]">{label}</span>
      <span className="min-w-0 flex-1 break-words text-white">{children}</span>
    </div>
  );
}

function RunActivityBarTooltip({
  run,
  nowMs,
}: {
  run: AutomationRun;
  nowMs: number;
}) {
  const { t, i18n } = useTranslation("openhands");
  const timestamp = getLastRunTimestamp(run);
  const when = timestamp
    ? formatRelativeTime(timestamp, i18n.language, t)
    : null;
  const durationMs = getAutomationRunDurationMs(run, nowMs);
  const durationLabel = formatDurationForTitle(durationMs);
  const statusLabel = t(getAutomationRunStatusLabelKey(run.status));
  const errorDetail =
    run.status === AutomationRunStatus.FAILED
      ? run.error_detail?.trim() || null
      : null;

  return (
    <div className="flex w-[220px] flex-col gap-2 p-3">
      <div className="flex items-center gap-2">
        <span
          aria-hidden="true"
          className={cn(
            "h-3 w-1.5 shrink-0 rounded-[1px]",
            barColorClassForStatus(run.status),
          )}
        />
        <span className="text-sm font-medium text-white">{statusLabel}</span>
      </div>

      <div className="flex flex-col gap-1">
        {when ? (
          <TooltipRow label={t(I18nKey.AUTOMATIONS$DETAIL$LAST_RUN)}>
            {when}
          </TooltipRow>
        ) : null}
        {durationLabel ? (
          <TooltipRow label={t(I18nKey.FEATURED_AUTOMATIONS$RUN_DURATION)}>
            {durationLabel}
          </TooltipRow>
        ) : null}
      </div>

      {errorDetail ? (
        <p className="line-clamp-3 text-xs text-[var(--oh-status-error)]">
          {errorDetail}
        </p>
      ) : null}
    </div>
  );
}

function RunActivityBar({
  automationId,
  run,
  nowMs,
}: {
  automationId: string;
  run: AutomationRun;
  nowMs: number;
}) {
  const { t } = useTranslation("openhands");
  const durationMs = getAutomationRunDurationMs(run, nowMs);
  const statusLabel = t(getAutomationRunStatusLabelKey(run.status));
  const disableAnimation = import.meta.env.MODE === "test";
  const href = `/automations/${encodeURIComponent(automationId)}?run=${encodeURIComponent(run.id)}`;

  return (
    <Tooltip
      content={<RunActivityBarTooltip run={run} nowMs={nowMs} />}
      placement="top"
      closeDelay={80}
      delay={200}
      disableAnimation={disableAnimation}
      className="rounded-xl border border-[var(--oh-border)] bg-base-secondary p-0 text-white shadow-xl"
    >
      {/* Wider hit target than the 4px bar so cancelled/skipped greys are easy to inspect. */}
      <NavigationLink
        to={href}
        aria-label={statusLabel}
        className="group/spark-bar inline-flex h-full cursor-pointer items-end px-[2px] focus:outline-none focus-visible:ring-1 focus-visible:ring-[var(--oh-focus)]"
        onMouseDown={(event) => event.stopPropagation()}
        onClick={(event) => event.stopPropagation()}
      >
        <span
          className={cn(
            "w-1 shrink-0 origin-bottom rounded-[1px]",
            "transition-transform duration-100 ease-out motion-reduce:transition-none",
            // Grow the hovered/focused bar so it reads clearly among neighbors.
            "group-hover/spark-bar:scale-x-[1.4] group-hover/spark-bar:scale-y-[1.15]",
            "group-focus-visible/spark-bar:scale-x-[1.4] group-focus-visible/spark-bar:scale-y-[1.15]",
            barColorClassForStatus(run.status),
          )}
          style={{ height: durationMsToSparklineBarHeightPx(durationMs) }}
        />
      </NavigationLink>
    </Tooltip>
  );
}

/**
 * Colored-bar sparkline of recent automation runs (oldest → newest left to
 * right). Color encodes status; height encodes run duration (log-scaled).
 * Bars link to the automation detail page focused on that run.
 */
export function AutomationRunActivitySparkline({
  automationId,
  runs,
  testId,
}: AutomationRunActivitySparklineProps) {
  const { t } = useTranslation("openhands");
  const hasInFlight = runs.some((run) => isInFlightAutomationRun(run));
  const [nowMs, setNowMs] = useState(() => Date.now());

  useEffect(() => {
    if (!hasInFlight) return undefined;
    // Tick while a run is in flight so bar height tracks elapsed time between
    // the slower runs-list polls.
    const id = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [hasInFlight]);

  if (runs.length === 0) return null;

  // API is newest-first; reverse so the sparkline reads left→right as time.
  const bars = [...runs].reverse().slice(-AUTOMATION_RUN_ACTIVITY_LIMIT);

  return (
    <span
      data-testid={testId}
      aria-label={t(I18nKey.FEATURED_AUTOMATIONS$RUN_ACTIVITY_LABEL, {
        count: bars.length,
      })}
      className="inline-flex shrink-0 items-end"
      style={{ height: AUTOMATION_RUN_ACTIVITY_TRACK_PX }}
      onMouseDown={(event) => event.preventDefault()}
    >
      {bars.map((run, index) => (
        <RunActivityBar
          key={`${run.id}-${index}`}
          automationId={automationId}
          run={run}
          nowMs={nowMs}
        />
      ))}
    </span>
  );
}
