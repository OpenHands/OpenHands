import { AlertCircle, ExternalLink, RotateCcw } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { AutomationHealthDetails } from "#/manifests/automation-insights";
import { formatAutomationFailureKind } from "#/manifests/automation-insights";
import type { Automation } from "#/types/automation";
import { cn } from "#/utils/utils";

interface AutomationHealthNoticeProps {
  automation: Pick<Automation, "id" | "enabled">;
  canManage: boolean;
  details: AutomationHealthDetails;
  onToggle: (id: string, enabled: boolean) => void;
  onView: () => void;
}

const ISSUE_STYLES = {
  disabled: {
    container: "border-[var(--oh-border)] bg-surface-raised text-muted",
    icon: RotateCcw,
  },
  blocked: {
    container:
      "border-[var(--oh-danger)]/40 bg-[var(--oh-danger)]/10 text-danger",
    icon: AlertCircle,
  },
  transient: {
    container:
      "border-[var(--oh-warning)]/40 bg-[var(--oh-warning)]/10 text-[var(--oh-warning)]",
    icon: AlertCircle,
  },
} as const;

export function AutomationHealthNotice({
  automation,
  canManage,
  details,
  onToggle,
  onView,
}: AutomationHealthNoticeProps) {
  const { t } = useTranslation("openhands");
  if (!details.issue || !details.reason) return null;

  const style = ISSUE_STYLES[details.issue];
  const Icon = style.icon;

  return (
    <div
      data-testid={`automation-health-notice-${automation.id}`}
      className={cn(
        "flex min-w-0 items-start gap-2 rounded-md border px-2.5 py-2 text-xs",
        style.container,
      )}
    >
      <Icon className="mt-0.5 size-3.5 shrink-0" aria-hidden="true" />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 font-medium">
          <span>
            {details.issue === "disabled"
              ? t(I18nKey.AUTOMATIONS$HEALTH_DISABLED)
              : details.issue === "blocked"
                ? t(I18nKey.AUTOMATIONS$HEALTH_BLOCKED)
                : t(I18nKey.AUTOMATIONS$HEALTH_TRANSIENT)}
          </span>
          {details.failureKind ? (
            <span className="font-normal opacity-80">
              ({formatAutomationFailureKind(details.failureKind)})
            </span>
          ) : null}
        </div>
        <p className="mt-0.5 break-words text-[11px] leading-relaxed text-content/80">
          {details.reason}
        </p>
        <div className="mt-1.5 flex flex-wrap items-center gap-2">
          {details.issue === "disabled" && canManage ? (
            <button
              type="button"
              className="font-medium underline underline-offset-2 hover:no-underline"
              onClick={(event) => {
                event.stopPropagation();
                onToggle(automation.id, automation.enabled);
              }}
            >
              {t(I18nKey.AUTOMATIONS$TURN_ON)}
            </button>
          ) : null}
          <button
            type="button"
            className="inline-flex items-center gap-1 font-medium underline underline-offset-2 hover:no-underline"
            onClick={(event) => {
              event.stopPropagation();
              onView();
            }}
          >
            <ExternalLink className="size-3" aria-hidden="true" />
            {t(I18nKey.COMMON$VIEW)}
          </button>
        </div>
      </div>
    </div>
  );
}
