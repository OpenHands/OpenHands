import type { LatestAutomationRunState } from "#/hooks/query/use-latest-automation-runs";
import { I18nKey } from "#/i18n/declaration";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
} from "#/types/automation";
import { formatEventOn } from "#/utils/automation-schedule";

export type AutomationRunHealth =
  | "success"
  | "failed"
  | "in_progress"
  | "none"
  | "unknown";

export function deriveRunHealth(
  state: LatestAutomationRunState,
): AutomationRunHealth {
  if (state.isLoading || state.isError) return "unknown";
  if (!state.latestRun) return "none";
  switch (state.latestRun.status) {
    case AutomationRunStatus.COMPLETED:
      return "success";
    case AutomationRunStatus.FAILED:
      return "failed";
    case AutomationRunStatus.PENDING:
    case AutomationRunStatus.RUNNING:
      return "in_progress";
    // CANCELLED, SKIPPED, and any status the backend adds after this enum.
    default:
      return "unknown";
  }
}

export function getRunHealthLabelKey(health: AutomationRunHealth): I18nKey {
  switch (health) {
    case "success":
      return I18nKey.FEATURED_AUTOMATIONS$LAST_RUN_SUCCEEDED;
    case "failed":
      return I18nKey.FEATURED_AUTOMATIONS$LAST_RUN_FAILED;
    case "in_progress":
      return I18nKey.FEATURED_AUTOMATIONS$RUN_IN_PROGRESS;
    case "none":
      return I18nKey.AUTOMATIONS$DETAIL$NO_RUNS;
    default:
      return I18nKey.FEATURED_AUTOMATIONS$STATUS_UNKNOWN;
  }
}

export function getTriggerSummary(automation: Automation): string {
  const { trigger } = automation;
  if (trigger.type === "event") {
    return [
      trigger.on ? formatEventOn(trigger.on) : "",
      trigger.source ? `(${trigger.source})` : "",
    ]
      .filter(Boolean)
      .join(" ");
  }
  return trigger.schedule_human || trigger.schedule || trigger.type;
}

/**
 * Timestamp to show as the run's "last run" moment, or null when the run
 * has no usable timestamp yet. The backend leaves started_at unset
 * (epoch/zero) while a run is PENDING and only populates it once execution
 * begins.
 */
export function getLastRunTimestamp(run: AutomationRun): string | null {
  const candidate = run.completed_at ?? run.started_at;
  if (!candidate) return null;
  const time = new Date(candidate).getTime();
  if (Number.isNaN(time) || time === 0) return null;
  return candidate;
}
