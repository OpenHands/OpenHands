import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";
import { CreateInstructions } from "./create-instructions";
import { RecommendedAutomationsLauncher } from "./recommended-automations-launcher";

export function EmptyState() {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="automations-empty"
      className={extensionModuleEmptyStateClassName}
    >
      <p className="text-sm text-white">{t(I18nKey.AUTOMATIONS$EMPTY)}</p>

      <div className="mt-4 flex justify-center">
        <CreateInstructions />
      </div>

      <div className="mt-8 w-full text-left">
        <RecommendedAutomationsLauncher className="pb-0" variant="rail" />
      </div>
    </div>
  );
}
