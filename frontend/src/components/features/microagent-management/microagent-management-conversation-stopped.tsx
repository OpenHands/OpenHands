import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "../settings/brand-button";
import { Loader } from "#/components/shared/loader";
import { useMicroagentManagementStore } from "#/state/microagent-management-store";

export function MicroagentManagementConversationStopped() {
  const { t } = useTranslation();

  const { selectedMicroagentItem } = useMicroagentManagementStore();

  const { conversation } = selectedMicroagentItem ?? {};

  const { conversation_id: conversationId } = conversation ?? {};

  if (!conversationId) {
    return null;
  }

  return (
    <div className="flex-1 flex flex-col h-full items-center justify-center">
      <div className="text-[#ffffff99] text-[22px] font-bold pb-[22px] text-center max-w-[455px]">
        {t(I18nKey.MICROAGENT_MANAGEMENT$CONVERSATION_STOPPED)}
      </div>
      <Loader size="small" className="pb-[22px]" />
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
    </div>
  );
}
