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

/** Known event sources rendered with brand casing; others title-case. */
const TRIGGER_SOURCE_LABELS: Record<string, string> = {
  github: "GitHub",
  gitlab: "GitLab",
  slack: "Slack",
  linear: "Linear",
  jira: "Jira",
};

export function formatTriggerSourceLabel(source: string): string {
  const key = source.trim().toLowerCase();
  if (!key) return source;
  return (
    TRIGGER_SOURCE_LABELS[key] ??
    key.replace(/\b\w/g, (char) => char.toUpperCase())
  );
}

export function getTriggerEventLabel(automation: Automation): string | null {
  const { trigger } = automation;
  if (trigger.type !== "event") return null;
  return trigger.on ? formatEventOn(trigger.on) : null;
}

export function getTriggerSource(automation: Automation): string | null {
  const { trigger } = automation;
  if (trigger.type !== "event") return null;
  return trigger.source?.trim() || null;
}

export function getTriggerScheduleLabel(automation: Automation): string | null {
  const { trigger } = automation;
  if (trigger.type === "event") return null;
  return trigger.schedule_human || trigger.schedule || trigger.type || null;
}

export function getTriggerSummary(automation: Automation): string {
  const { trigger } = automation;
  if (trigger.type === "event") {
    const source = getTriggerSource(automation);
    return [
      getTriggerEventLabel(automation) ?? "",
      source ? `(${formatTriggerSourceLabel(source)})` : "",
    ]
      .filter(Boolean)
      .join(" ");
  }
  return getTriggerScheduleLabel(automation) ?? trigger.type;
}

/** Inline status-row preview length (keeps the status area to one line). */
const ERROR_DETAIL_INLINE_MAX = 48;

/**
 * Short preview for the pinned-card status row. Prefer the first sentence,
 * then hard-truncate so the row stays single-line.
 */
export function shortenAutomationErrorDetail(detail: string): string {
  const trimmed = detail.trim().replace(/\s+/g, " ");
  if (!trimmed) return trimmed;
  const firstSentence = trimmed.match(/^.*?[.!?](?=\s|$)/)?.[0] ?? trimmed;
  if (firstSentence.length <= ERROR_DETAIL_INLINE_MAX) return firstSentence;
  return `${firstSentence.slice(0, ERROR_DETAIL_INLINE_MAX - 1).trimEnd()}…`;
}

/** True when the inline preview is shortened and the full message belongs in a hovercard. */
export function shouldShowAutomationErrorHovercard(
  detail: string,
  shortDetail: string = shortenAutomationErrorDetail(detail),
): boolean {
  return detail.trim().replace(/\s+/g, " ") !== shortDetail;
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
