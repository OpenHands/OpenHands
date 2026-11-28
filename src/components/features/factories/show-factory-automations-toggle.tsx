import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { ToggleSwitch } from "#/components/features/automations/toggle-switch";

interface ShowFactoryAutomationsToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
}

/** Whether the automation list also shows the automations that are in a factory. */
export function ShowFactoryAutomationsToggle({
  checked,
  onChange,
}: ShowFactoryAutomationsToggleProps) {
  const { t } = useTranslation("openhands");
  const label = t(I18nKey.AUTOMATIONS$SHOW_FACTORY_AUTOMATIONS);

  return (
    <div
      data-testid="automations-show-factory-automations"
      className="flex h-9 shrink-0 items-center gap-2 rounded-lg border border-border bg-base-secondary px-3 text-sm text-contrast"
    >
      <span className="whitespace-nowrap">{label}</span>
      <ToggleSwitch
        enabled={checked}
        label={label}
        onToggle={() => onChange(!checked)}
      />
    </div>
  );
}
