import { useState } from "react";
import { Server } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";
import { useUpdateAutomationKeepAlive } from "#/hooks/query/use-automations";
import { ToggleSwitch } from "#/components/features/automations/toggle-switch";
import ChevronDownIcon from "#/icons/chevron-down.svg?react";
import { isKeepAliveForced } from "#/utils/automation-state";
import { cn } from "#/utils/utils";

interface KeepAliveSectionProps {
  automation: Automation;
  canEdit: boolean;
}

/**
 * The "Advanced" disclosure on the detail page, collapsed by default. It shows
 * and saves `keep_alive`, the sandbox cleanup policy after each run.
 */
export function KeepAliveSection({
  automation,
  canEdit,
}: KeepAliveSectionProps) {
  const { t } = useTranslation("openhands");
  const updateKeepAlive = useUpdateAutomationKeepAlive();
  const [isExpanded, setIsExpanded] = useState(false);

  // An automation service that does not return the field would reject it in
  // a PATCH.
  if (automation.keep_alive === undefined) return null;

  const isForced = isKeepAliveForced(automation);
  const savedValue = automation.keep_alive === true;
  const enabled =
    isForced ||
    (updateKeepAlive.isPending
      ? updateKeepAlive.variables.keepAlive
      : savedValue);

  return (
    <div className="rounded-2xl border border-border bg-surface">
      <button
        type="button"
        data-testid="automation-advanced-toggle"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
        className={cn(
          "flex w-full items-center gap-2 rounded-2xl px-5 py-3 text-left transition-colors hover:bg-surface-raised",
          isExpanded && "rounded-b-none border-b border-border",
        )}
      >
        <Server className="size-4 text-muted" aria-hidden="true" />
        <h3 className="text-sm font-medium text-content">
          {t(I18nKey.SETTINGS$ADVANCED)}
        </h3>
        <ChevronDownIcon
          aria-hidden="true"
          className={cn(
            "ml-auto size-5 text-muted transition-transform",
            isExpanded && "rotate-180",
          )}
        />
      </button>
      {isExpanded ? (
        <div
          data-testid="automation-keep-alive"
          className="flex items-start justify-between gap-4 px-5 py-5"
        >
          <div className="flex min-w-0 flex-col gap-1">
            <span className="text-sm text-content">
              {t(I18nKey.AUTOMATIONS$DETAIL$KEEP_ALIVE)}
            </span>
            <p className="text-xs text-muted">
              {isForced
                ? t(I18nKey.AUTOMATIONS$DETAIL$KEEP_ALIVE_FORCED)
                : t(I18nKey.AUTOMATIONS$DETAIL$KEEP_ALIVE_DESCRIPTION)}
            </p>
          </div>
          <ToggleSwitch
            enabled={enabled}
            label={t(I18nKey.AUTOMATIONS$DETAIL$KEEP_ALIVE)}
            disabled={isForced || !canEdit || updateKeepAlive.isPending}
            onToggle={() =>
              updateKeepAlive.mutate({
                id: automation.id,
                keepAlive: !savedValue,
              })
            }
            className="disabled:cursor-not-allowed disabled:opacity-60"
          />
        </div>
      ) : null}
    </div>
  );
}
