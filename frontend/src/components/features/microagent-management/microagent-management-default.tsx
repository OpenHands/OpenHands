import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

export function MicroagentManagementDefault() {
  const { t } = useTranslation();

  return (
    <div className="flex-1 flex flex-col h-full items-center justify-center">
      <div className="text-content text-xl font-bold pb-4">
        {t(I18nKey.MICROAGENT_MANAGEMENT$READY_TO_ADD_MICROAGENT)}
      </div>
      <div className="text-content text-sm font-normal text-center max-w-[455px]">
        {t(
          I18nKey.MICROAGENT_MANAGEMENT$OPENHANDS_CAN_LEARN_ABOUT_REPOSITORIES,
        )}
      </div>
    </div>
  );
}
