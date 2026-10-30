import { Server } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";
import { useUpdateAutomationKeepAlive } from "#/hooks/query/use-automations";
import { ToggleSwitch } from "#/components/features/automations/toggle-switch";
import { isKeepAliveForced } from "#/utils/automation-state";
import { SectionCard } from "./section-card";

interface KeepAliveSectionProps {
  automation: Automation;
  canEdit: boolean;
}

/** Shows and saves `keep_alive`, the sandbox cleanup policy after each run. */
export function KeepAliveSection({
  automation,
  canEdit,
}: KeepAliveSectionProps) {
  const { t } = useTranslation("openhands");
  const updateKeepAlive = useUpdateAutomationKeepAlive();

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
    <SectionCard
      icon={<Server className="size-4" aria-hidden="true" />}
      title={t(I18nKey.SETTINGS$ADVANCED)}
    >
      <div
        data-testid="automation-keep-alive"
        className="flex items-start justify-between gap-4"
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
    </SectionCard>
  );
}
