import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";
import {
  getAutomationLifecycleState,
  type AutomationLifecycleState,
} from "#/utils/automation-state";
import { cn } from "#/utils/utils";

interface ActiveStatusBadgeProps {
  automation: Pick<Automation, "enabled" | "state">;
  compact?: boolean;
}

const STATUS_DISPLAY: Record<
  AutomationLifecycleState,
  { testId: string; label: I18nKey; className: string }
> = {
  active: {
    testId: "active-status-badge-active",
    label: I18nKey.AUTOMATIONS$DETAIL$ACTIVE,
    className: "bg-semantic-success/15 text-semantic-success",
  },
  inactive: {
    testId: "active-status-badge-inactive",
    label: I18nKey.AUTOMATIONS$DETAIL$INACTIVE,
    className: "bg-surface-raised text-muted",
  },
  draft: {
    testId: "active-status-badge-draft",
    label: I18nKey.AUTOMATIONS$DETAIL$DRAFT,
    className: "bg-[var(--oh-warning)]/15 text-[var(--oh-warning)]",
  },
};

export function ActiveStatusBadge({
  automation,
  compact = false,
}: ActiveStatusBadgeProps) {
  const { t } = useTranslation("openhands");
  const display = STATUS_DISPLAY[getAutomationLifecycleState(automation)];

  return (
    <span
      data-testid={display.testId}
      className={cn(
        "inline-flex items-center rounded-full font-medium",
        compact ? "px-2 py-0.5 text-[10px]" : "px-3 py-1 text-xs",
        display.className,
      )}
    >
      {t(display.label)}
    </span>
  );
}
