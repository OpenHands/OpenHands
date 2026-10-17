import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import ClockIcon from "#/icons/clock.svg?react";
import {
  formatCompactDuration,
  type RunSummaryState,
} from "#/manifests/automation-insights";
import type { InterfaceListInsights } from "#/manifests/types";

export function runCountDisplay(state: RunSummaryState | undefined): string {
  if (state?.summary) return state.summary.total.toLocaleString();
  if (!state || state.isLoading) return "…";
  return "—";
}

export function successRateDisplay(state: RunSummaryState | undefined): string {
  const rate = state?.summary?.recentSuccessRate;
  if (rate === null || rate === undefined) return "—";
  return `${Math.round(rate * 100)}%`;
}

export function averageDurationDisplay(
  state: RunSummaryState | undefined,
): string {
  return formatCompactDuration(state?.summary?.averageDurationMs ?? null);
}

/**
 * Queued-run count worth surfacing: the summary's lifetime PENDING count, or
 * null when there is nothing queued — or when the count is unknown (older
 * service, partial sample) and showing a number would be a guess.
 */
export function pendingRunsDisplay(
  state: RunSummaryState | undefined,
): number | null {
  const pending = state?.summary?.pendingTotal;
  return typeof pending === "number" && pending > 0 ? pending : null;
}

interface AutomationPendingCountProps {
  state: RunSummaryState | undefined;
}

/**
 * "N Pending" queue indicator for an automation card/row. Renders nothing
 * unless at least one run is queued: a permanent zero would be noise on
 * every automation that is not backed up.
 */
export function AutomationPendingCount({ state }: AutomationPendingCountProps) {
  const { t } = useTranslation("openhands");
  const pending = pendingRunsDisplay(state);
  if (pending === null) return null;
  return (
    <span
      data-testid="automation-pending-count"
      className="inline-flex shrink-0 items-center gap-1 text-xs font-medium text-muted"
    >
      <ClockIcon className="size-3.5" aria-hidden="true" />
      {pending} {t(I18nKey.AUTOMATIONS$DETAIL$PENDING)}
    </span>
  );
}

interface AutomationRunStatsProps {
  state: RunSummaryState | undefined;
  copy: InterfaceListInsights["stats"];
}

/** The three-column run stats footer on an automation card. */
export function AutomationRunStats({ state, copy }: AutomationRunStatsProps) {
  const cells = [
    { label: copy.runs, value: runCountDisplay(state) },
    { label: copy.recentSuccess, value: successRateDisplay(state) },
    { label: copy.averageDuration, value: averageDurationDisplay(state) },
  ];
  return (
    <dl
      data-testid="automation-run-stats"
      className="grid grid-cols-3 gap-2 text-xs"
    >
      {cells.map((cell) => (
        <div key={cell.label} className="min-w-0">
          <dt className="truncate text-muted">{cell.label}</dt>
          <dd className="mt-0.5 truncate font-medium text-content">
            {cell.value}
          </dd>
        </div>
      ))}
    </dl>
  );
}
