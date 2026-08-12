import { I18nKey } from "#/i18n/declaration";
import { AutomationRunStatus, type AutomationRun } from "#/types/automation";

/** Sparkline track height; bars sit on the baseline inside this. */
export const AUTOMATION_RUN_ACTIVITY_TRACK_PX = 18;
export const AUTOMATION_RUN_ACTIVITY_MIN_BAR_PX = 6;
export const AUTOMATION_RUN_ACTIVITY_MAX_BAR_PX = 15;
/** Used when duration cannot be derived (no timestamps, skipped, etc.). */
export const AUTOMATION_RUN_ACTIVITY_UNKNOWN_BAR_PX = 11;

const MIN_DURATION_MS = 1_000;
/** Log-scale ceiling so multi-hour runs still fit the track. */
const MAX_DURATION_MS = 30 * 60_000;

/**
 * Wall-clock duration for a run, or null when it cannot be measured.
 * In-flight runs use elapsed time since `started_at` when that is usable.
 */
export function getAutomationRunDurationMs(
  run: AutomationRun,
  nowMs: number = Date.now(),
): number | null {
  const startMs = run.started_at ? new Date(run.started_at).getTime() : NaN;
  if (Number.isNaN(startMs) || startMs <= 0) return null;

  if (run.completed_at) {
    const endMs = new Date(run.completed_at).getTime();
    if (Number.isNaN(endMs) || endMs < startMs) return null;
    return endMs - startMs;
  }

  if (
    run.status === AutomationRunStatus.PENDING ||
    run.status === AutomationRunStatus.RUNNING
  ) {
    return Math.max(0, nowMs - startMs);
  }

  return null;
}

/** Map duration → bar height (log scale, clamped). Color stays status-driven. */
export function durationMsToSparklineBarHeightPx(
  durationMs: number | null,
): number {
  if (durationMs == null || durationMs <= 0) {
    return AUTOMATION_RUN_ACTIVITY_UNKNOWN_BAR_PX;
  }

  const clamped = Math.min(
    MAX_DURATION_MS,
    Math.max(MIN_DURATION_MS, durationMs),
  );
  const t =
    Math.log(clamped / MIN_DURATION_MS) /
    Math.log(MAX_DURATION_MS / MIN_DURATION_MS);

  return Math.round(
    AUTOMATION_RUN_ACTIVITY_MIN_BAR_PX +
      t *
        (AUTOMATION_RUN_ACTIVITY_MAX_BAR_PX -
          AUTOMATION_RUN_ACTIVITY_MIN_BAR_PX),
  );
}

export function barColorClassForStatus(
  status: AutomationRunStatus | string,
): string {
  switch (status) {
    case AutomationRunStatus.COMPLETED:
      return "bg-[var(--oh-status-success)]";
    case AutomationRunStatus.FAILED:
      return "bg-[var(--oh-status-error)]";
    case AutomationRunStatus.RUNNING:
      return "bg-[var(--oh-status-success)] animate-pulse motion-reduce:animate-none";
    case AutomationRunStatus.PENDING:
      // Grey hatch so pending reads as queued, distinct from solid cancelled/skipped.
      return "bg-[repeating-linear-gradient(-45deg,var(--oh-muted)_0_1.5px,var(--oh-border)_1.5px_3px)]";
    case AutomationRunStatus.CANCELLED:
      return "bg-[var(--oh-muted)]";
    default:
      return "bg-[var(--oh-border)]";
  }
}

export function getAutomationRunStatusLabelKey(
  status: AutomationRunStatus | string,
): I18nKey {
  switch (status) {
    case AutomationRunStatus.COMPLETED:
      return I18nKey.AUTOMATIONS$DETAIL$SUCCESSFUL;
    case AutomationRunStatus.FAILED:
      return I18nKey.AUTOMATIONS$DETAIL$FAILED;
    case AutomationRunStatus.PENDING:
      return I18nKey.AUTOMATIONS$DETAIL$PENDING;
    case AutomationRunStatus.RUNNING:
      return I18nKey.AUTOMATIONS$DETAIL$RUNNING;
    case AutomationRunStatus.CANCELLED:
      return I18nKey.AUTOMATIONS$DETAIL$CANCELLED;
    case AutomationRunStatus.SKIPPED:
      return I18nKey.AUTOMATIONS$DETAIL$SKIPPED;
    default:
      return I18nKey.FEATURED_AUTOMATIONS$STATUS_UNKNOWN;
  }
}

export function formatDurationForTitle(
  durationMs: number | null,
): string | null {
  if (durationMs == null || durationMs <= 0) return null;
  if (durationMs < 60_000) {
    return `${Math.max(1, Math.round(durationMs / 1000))}s`;
  }
  if (durationMs < 3_600_000) {
    return `${Math.round(durationMs / 60_000)}m`;
  }
  return `${(durationMs / 3_600_000).toFixed(1)}h`;
}
