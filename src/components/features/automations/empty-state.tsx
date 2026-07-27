import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";
import { CreateInstructions } from "./create-instructions";

export function EmptyState() {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="automations-empty"
      className={extensionModuleEmptyStateClassName}
    >
      <p className="text-sm text-white">{t(I18nKey.AUTOMATIONS$EMPTY)}</p>
      <p className="mt-1 text-xs text-tertiary-light">
        {t(I18nKey.AUTOMATIONS$EMPTY_HINT)}
      </p>

      <div className="mt-8 flex justify-center border-t border-[var(--oh-border)] pt-8">
        <CreateInstructions />
      </div>
    </div>
  );
}
