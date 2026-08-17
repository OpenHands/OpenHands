import {
  formatCompactDuration,
  type RunSummaryState,
} from "#/manifests/automation-insights";
import type { InterfaceListInsights } from "#/manifests/types";
import { formatTimeDelta } from "#/utils/format-time-delta";

/** `formatTimeDelta` renders seconds as `<n>s`; anything under a minute reads as "just now". */
const SECONDS_DELTA_PATTERN = /^\d+s$/;

/**
 * "Never", "Just now", or a localized "<delta> ago". The captions are the
 * manifest's; the delta and its "ago" suffix are the host's translations.
 */
export function lastRunText(
  startedAt: string | null | undefined,
  copy: InterfaceListInsights["lastRun"],
  agoSuffix: string,
): string {
  if (!startedAt) return copy.never;
  const delta = formatTimeDelta(startedAt);
  return SECONDS_DELTA_PATTERN.test(delta)
    ? copy.justNow
    : `${delta} ${agoSuffix}`;
}

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
      className="grid grid-cols-3 gap-2 border-t border-[var(--oh-border)] pt-3 text-xs"
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
