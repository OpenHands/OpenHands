import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface ActiveStatusBadgeProps {
  active: boolean;
}

export function ActiveStatusBadge({ active }: ActiveStatusBadgeProps) {
  const { t } = useTranslation("openhands");

  return (
    <span
      data-testid={
        active ? "active-status-badge-active" : "active-status-badge-inactive"
      }
      className={cn(
        "inline-flex items-center rounded-full px-3 py-1 text-xs font-medium",
        active
          ? "bg-semantic-success/15 text-semantic-success"
          : "bg-surface-raised text-muted",
      )}
    >
      {active
        ? t(I18nKey.AUTOMATIONS$DETAIL$ACTIVE)
        : t(I18nKey.AUTOMATIONS$DETAIL$INACTIVE)}
    </span>
  );
}
