import { Loader2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { I18nKey } from "#/i18n/declaration";
import CheckCircleIcon from "#/icons/check-circle.svg?react";
import XCircleIcon from "#/icons/x-circle.svg?react";
import ClockIcon from "#/icons/clock.svg?react";
import { AutomationRunStatus } from "#/types/automation";
import { cn } from "#/utils/utils";

interface RunStatusBadgeProps {
  status: AutomationRunStatus;
  /**
   * Icon-only mark (no pill). Status text is available via hover tooltip and
   * `aria-label`, unless {@link showLabel} is also set.
   */
  iconOnly?: boolean;
  /** With `iconOnly`, render the status word next to the icon (still no pill). */
  showLabel?: boolean;
  /** Smaller pill for dense rows (e.g. home activity meta line). */
  compact?: boolean;
}

const statusConfig: Record<
  AutomationRunStatus,
  { label: I18nKey; style: string; iconTone: string }
> = {
  [AutomationRunStatus.COMPLETED]: {
    label: I18nKey.AUTOMATIONS$DETAIL$SUCCESSFUL,
    style:
      "border-[var(--oh-success)]/50 bg-[var(--oh-success)]/10 text-[var(--oh-success)]",
    iconTone: "text-[var(--oh-success)]",
  },
  [AutomationRunStatus.FAILED]: {
    label: I18nKey.AUTOMATIONS$DETAIL$FAILED,
    style: "border-[var(--oh-danger)]/50 bg-[var(--oh-danger)]/10 text-danger",
    iconTone: "text-danger",
  },
  [AutomationRunStatus.PENDING]: {
    label: I18nKey.AUTOMATIONS$DETAIL$PENDING,
    style: "border-[var(--oh-border)] bg-surface-raised text-muted",
    iconTone: "text-muted",
  },
  [AutomationRunStatus.RUNNING]: {
    label: I18nKey.AUTOMATIONS$DETAIL$RUNNING,
    style: "border-[var(--oh-border)] bg-surface-raised text-muted",
    iconTone: "text-muted",
  },
  [AutomationRunStatus.CANCELLED]: {
    label: I18nKey.AUTOMATIONS$DETAIL$CANCELLED,
    style: "border-[var(--oh-border)] bg-surface-raised text-muted",
    iconTone: "text-muted",
  },
  [AutomationRunStatus.SKIPPED]: {
    label: I18nKey.AUTOMATIONS$DETAIL$SKIPPED,
    style: "border-[var(--oh-border)] bg-surface-raised text-muted",
    iconTone: "text-muted",
  },
};

function StatusIcon({
  status,
  compact = false,
}: {
  status: AutomationRunStatus;
  compact?: boolean;
}) {
  const iconClass = compact ? "size-3" : "size-3.5";
  switch (status) {
    case AutomationRunStatus.COMPLETED:
      return (
        <CheckCircleIcon
          data-testid="run-status-icon-completed"
          className={iconClass}
        />
      );
    case AutomationRunStatus.FAILED:
      return (
        <XCircleIcon
          data-testid="run-status-icon-failed"
          className={iconClass}
        />
      );
    case AutomationRunStatus.RUNNING:
      return (
        <Loader2
          data-testid="run-status-icon-running"
          className={cn(iconClass, "animate-spin motion-reduce:animate-none")}
          aria-hidden="true"
        />
      );
    default:
      return (
        <ClockIcon
          data-testid="run-status-icon-pending"
          className={iconClass}
        />
      );
  }
}

export function RunStatusBadge({
  status,
  iconOnly = false,
  showLabel = false,
  compact = false,
}: RunStatusBadgeProps) {
  const { t } = useTranslation("openhands");
  // Degrade instead of crashing on a status the backend added after this enum.
  const config =
    statusConfig[status] ?? statusConfig[AutomationRunStatus.PENDING];
  const label = t(config.label);

  if (iconOnly) {
    if (showLabel) {
      return (
        <span
          data-testid="run-status-badge-icon"
          className={cn(
            "inline-flex min-w-0 shrink-0 items-center gap-1.5 text-xs font-medium",
            config.iconTone,
          )}
        >
          <StatusIcon status={status} compact={compact} />
          <span className="truncate">{label}</span>
        </span>
      );
    }

    return (
      <StyledTooltip content={label} placement="top">
        <span
          role="img"
          aria-label={label}
          data-testid="run-status-badge-icon"
          className={cn(
            "inline-flex shrink-0 cursor-default items-center justify-center",
            config.iconTone,
          )}
        >
          <StatusIcon status={status} compact={compact} />
        </span>
      </StyledTooltip>
    );
  }

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border font-medium",
        compact
          ? "gap-1 px-1.5 py-0 text-[10px] leading-4"
          : "gap-1.5 px-2.5 py-1 text-xs",
        config.style,
      )}
    >
      <StatusIcon status={status} compact={compact} />
      {label}
    </span>
  );
}
