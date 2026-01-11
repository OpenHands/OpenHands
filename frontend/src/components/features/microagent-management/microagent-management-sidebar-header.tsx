import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import QuestionCircleIcon from "#/icons/question-circle.svg?react";
import { DOCUMENTATION_URL } from "#/utils/constants";

export function MicroagentManagementSidebarHeader() {
  const { t } = useTranslation();

  return (
    <div>
      <h1 className="text-content text-[28px] font-bold">
        {t(I18nKey.MICROAGENT_MANAGEMENT$DESCRIPTION)}
      </h1>
      <p className="text-content text-sm font-normal leading-[20px] pt-2">
        {t(I18nKey.MICROAGENT_MANAGEMENT$USE_MICROAGENTS)}
        <a
          href={DOCUMENTATION_URL.MICROAGENTS.MICROAGENTS_OVERVIEW}
          target="_blank"
          rel="noopener noreferrer"
        >
          <QuestionCircleIcon className="inline-block ml-1" />
        </a>
      </p>
    </div>
  );
}
