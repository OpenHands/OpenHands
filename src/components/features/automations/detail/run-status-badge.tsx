import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import CheckCircleIcon from "#/icons/check-circle.svg?react";
import XCircleIcon from "#/icons/x-circle.svg?react";
import ClockIcon from "#/icons/clock.svg?react";
import { AutomationRunStatus } from "#/types/automation";
import { cn } from "#/utils/utils";

interface RunStatusBadgeProps {
  status: AutomationRunStatus;
}

const statusConfig: Record<
  AutomationRunStatus,
  { label: I18nKey; style: string }
> = {
  [AutomationRunStatus.COMPLETED]: {
    label: I18nKey.AUTOMATIONS$DETAIL$SUCCESSFUL,
    style:
      "border-[var(--oh-success)]/50 bg-[var(--oh-success)]/10 text-[var(--oh-success)]",
  },
  [AutomationRunStatus.FAILED]: {
    label: I18nKey.AUTOMATIONS$DETAIL$FAILED,
    style: "border-[var(--oh-danger)]/50 bg-[var(--oh-danger)]/10 text-danger",
  },
  [AutomationRunStatus.PENDING]: {
    label: I18nKey.AUTOMATIONS$DETAIL$PENDING,
    style: "border-[var(--oh-border)] bg-surface-raised text-muted",
  },
  [AutomationRunStatus.RUNNING]: {
    label: I18nKey.AUTOMATIONS$DETAIL$RUNNING,
    style: "border-[var(--oh-border)] bg-surface-raised text-muted",
  },
};

function StatusIcon({ status }: { status: AutomationRunStatus }) {
  switch (status) {
    case AutomationRunStatus.COMPLETED:
      return (
        <CheckCircleIcon
          data-testid="run-status-icon-completed"
          className="size-3.5"
        />
      );
    case AutomationRunStatus.FAILED:
      return (
        <XCircleIcon
          data-testid="run-status-icon-failed"
          className="size-3.5"
        />
      );
    default:
      return (
        <ClockIcon data-testid="run-status-icon-pending" className="size-3.5" />
      );
  }
}

export function RunStatusBadge({ status }: RunStatusBadgeProps) {
  const { t } = useTranslation("openhands");
  const config = statusConfig[status];

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium",
        config.style,
      )}
    >
      <StatusIcon status={status} />
      {t(config.label)}
    </span>
  );
}
