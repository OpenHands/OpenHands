import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { TaskOutcome, TaskOutcomeStatus } from "#/types/automation";

interface TaskOutcomeSummaryProps {
  outcome: TaskOutcome;
}

function getStatusLabelKey(status: TaskOutcomeStatus): I18nKey {
  switch (status) {
    case "success":
      return I18nKey.AUTOMATIONS$DETAIL$SUCCESSFUL;
    case "partial_success":
      return I18nKey.COMMON$COMPLETED_PARTIALLY;
    case "blocked":
      return I18nKey.AUTOMATIONS$DETAIL$TASK_OUTCOME_BLOCKED;
    case "failed":
      return I18nKey.AUTOMATIONS$DETAIL$FAILED;
    case "unknown":
      return I18nKey.COMMON$UNKNOWN;
  }
}

function getStatusTextClassName(status: TaskOutcomeStatus): string {
  switch (status) {
    case "success":
      return "text-muted";
    case "partial_success":
      return "text-[var(--oh-primary)]";
    case "blocked":
    case "failed":
      return "text-[var(--oh-status-error)]";
    case "unknown":
      return "text-muted";
  }
}

export function TaskOutcomeSummary({ outcome }: TaskOutcomeSummaryProps) {
  const { t } = useTranslation("openhands");
  const blockerMessages = outcome.blockers.map((blocker) => blocker.message);
  const showStatusText = outcome.status !== "success";

  if (
    !showStatusText &&
    !outcome.outcome_summary &&
    !outcome.needs_user_action &&
    blockerMessages.length === 0
  ) {
    return null;
  }

  return (
    <div
      data-testid="task-outcome-summary"
      className="min-w-0 space-y-1 text-xs"
    >
      <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
        <span className="text-muted">
          {t(I18nKey.AUTOMATIONS$DETAIL$TASK_OUTCOME)}
        </span>
        {showStatusText && (
          <span
            className={`font-medium ${getStatusTextClassName(outcome.status)}`}
          >
            {t(getStatusLabelKey(outcome.status))}
          </span>
        )}
        {outcome.needs_user_action && (
          <span className="font-medium text-[var(--oh-primary)]">
            {t(I18nKey.AUTOMATIONS$DETAIL$NEEDS_USER_ACTION)}
          </span>
        )}
        {outcome.outcome_summary && (
          <span className="min-w-0 max-w-full truncate text-muted">
            {outcome.outcome_summary}
          </span>
        )}
      </div>
      {blockerMessages.length > 0 && (
        <div className="min-w-0 max-w-full truncate text-[var(--oh-status-error)]">
          {blockerMessages.join(" · ")}
        </div>
      )}
    </div>
  );
}
