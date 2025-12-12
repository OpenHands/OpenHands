import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "../settings/brand-button";
import { getProviderName, constructPullRequestUrl } from "#/utils/utils";
import { Provider } from "#/types/settings";
import { useMicroagentManagementStore } from "#/state/microagent-management-store";

export function MicroagentManagementReviewPr() {
  const { t } = useTranslation();

  const { selectedMicroagentItem } = useMicroagentManagementStore();

  const { conversation } = selectedMicroagentItem ?? {};

  const {
    conversation_id: conversationId,
    selected_repository: selectedRepository,
    git_provider: gitProvider,
    pr_number: prNumber,
  } = conversation ?? {};

  if (!conversationId) {
    return null;
  }

  return (
    <div className="flex-1 flex flex-col h-full items-center justify-center">
      <div className="text-[#ffffff99] text-[22px] font-bold pb-[22px] text-center max-w-[455px]">
        {t(I18nKey.MICROAGENT_MANAGEMENT$YOUR_MICROAGENT_IS_READY)}
      </div>
      <div className="flex gap-[22px]">
        <a
          href={`/conversations/${conversationId}`}
          target="_blank"
          rel="noopener noreferrer"
        >
          <BrandButton
            type="button"
            variant="secondary"
            testId="view-conversation-button"
          >
            {t(I18nKey.MICROAGENT$VIEW_CONVERSATION)}
          </BrandButton>
        </a>
        <a
          href={
            selectedRepository && gitProvider && prNumber && prNumber.length > 0
              ? constructPullRequestUrl(
                  prNumber[0],
                  gitProvider,
                  selectedRepository,
                )
              : "/#"
          }
          target="_blank"
          rel="noopener noreferrer"
        >
          <BrandButton
            type="button"
            variant="primary"
            testId="view-conversation-button"
          >
            {`${t(I18nKey.COMMON$REVIEW_PR_IN)} ${getProviderName(
              gitProvider as Provider,
            )}`}
          </BrandButton>
        </a>
      </div>
    </div>
  );
}
