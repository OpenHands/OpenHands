import type { LatestAutomationRunState } from "#/hooks/query/use-latest-automation-runs";
import { I18nKey } from "#/i18n/declaration";
import { AutomationRunStatus, type Automation } from "#/types/automation";
import { formatRelativeTime } from "#/utils/format-relative-time";
import {
  getLastRunTimestamp,
  getTriggerSummary,
} from "./automation-run-health";

/** Live home-page row/card model derived from an automation + its latest run. */
export interface HomeAutomationActivityItem {
  id: string;
  name: string;
  triggerSummary: string;
  /** Null when the automation has never run (or the run fetch failed). */
  status: AutomationRunStatus | null;
  /** Relative time label, or null when there is no usable timestamp. */
  whenLabel: string | null;
  conversationId: string | null;
}

type Translate = (key: I18nKey, options?: Record<string, unknown>) => string;

export function buildHomeAutomationActivityItem(
  automation: Automation,
  runState: LatestAutomationRunState,
  locale: string,
  t: Translate,
): HomeAutomationActivityItem {
  const { latestRun } = runState;
  const timestamp = latestRun ? getLastRunTimestamp(latestRun) : null;

  return {
    id: automation.id,
    name: automation.name,
    triggerSummary: getTriggerSummary(automation),
    status: latestRun?.status ?? null,
    whenLabel: timestamp ? formatRelativeTime(timestamp, locale, t) : null,
    conversationId: latestRun?.conversation_id ?? null,
  };
}

function activitySortScore(
  automationId: string,
  runStates: Map<string, LatestAutomationRunState>,
): number {
  const state = runStates.get(automationId);
  const status = state?.latestRun?.status;
  if (
    status === AutomationRunStatus.PENDING ||
    status === AutomationRunStatus.RUNNING
  ) {
    return Number.MAX_SAFE_INTEGER;
  }
  const timestamp = state?.latestRun
    ? getLastRunTimestamp(state.latestRun)
    : null;
  return timestamp ? new Date(timestamp).getTime() : 0;
}

/**
 * Map enabled automations + latest-run state into sorted activity rows
 * (in-flight first, then newest last-run timestamp).
 */
export function buildHomeAutomationActivityItems(
  automations: readonly Automation[],
  runStates: Map<string, LatestAutomationRunState>,
  locale: string,
  t: Translate,
  unknownRunState: LatestAutomationRunState,
): HomeAutomationActivityItem[] {
  return [...automations]
    .map((automation) =>
      buildHomeAutomationActivityItem(
        automation,
        runStates.get(automation.id) ?? unknownRunState,
        locale,
        t,
      ),
    )
    .sort(
      (a, b) =>
        activitySortScore(b.id, runStates) - activitySortScore(a.id, runStates),
    );
}

export function hrefForActivityItem(item: HomeAutomationActivityItem): string {
  return item.conversationId
    ? `/conversations/${item.conversationId}`
    : `/automations/${item.id}`;
}
