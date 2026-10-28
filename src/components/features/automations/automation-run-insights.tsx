import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
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

interface AutomationPendingRunCountProps {
  automationId: string;
  state: RunSummaryState | undefined;
}

/** The automation's queued-run count; nothing when none or unknown. */
export function AutomationPendingRunCount({
  automationId,
  state,
}: AutomationPendingRunCountProps) {
  const { t } = useTranslation("openhands");
  const count = state?.summary?.pendingTotal;
  if (!count) return null;
  return (
    <span
      data-testid={`automation-pending-${automationId}`}
      className="shrink-0 text-text-secondary"
    >
      {t(I18nKey.AUTOMATIONS$PENDING_RUN_COUNT, {
        count: count.toLocaleString(),
      })}
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
