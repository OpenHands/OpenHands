import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";

export function FactoryAutomationsEmptyState() {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="factory-automations-empty"
      className={extensionModuleEmptyStateClassName}
    >
      <p className="text-sm text-contrast">
        {t(I18nKey.FACTORIES$AUTOMATIONS_EMPTY)}
      </p>
    </div>
  );
}
