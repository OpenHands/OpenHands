import { FaArchive } from "react-icons/fa";
import { useTranslation } from "react-i18next";
import { ConversationStatus } from "#/types/conversation-status";
import { I18nKey } from "#/i18n/declaration";
import CircleErrorIcon from "#/icons/circle-error.svg?react";

interface ConversationStatusBadgesProps {
  conversationStatus: ConversationStatus;
}

export function ConversationStatusBadges({
  conversationStatus,
}: ConversationStatusBadgesProps) {
  const { t } = useTranslation();

  if (conversationStatus === "ARCHIVED") {
    return (
      <span className="flex items-center gap-1 px-1.5 py-0.5 bg-[#868E96] text-white text-xs font-medium rounded-full opacity-60">
        <FaArchive size={10} className="text-white" />
        <span>{t(I18nKey.COMMON$ARCHIVED)}</span>
      </span>
    );
  }

  if (conversationStatus === "ERROR") {
    return (
      <span className="flex items-center gap-1 px-1.5 py-0.5 bg-[#FF684E] text-white text-xs font-medium rounded-full">
        <CircleErrorIcon className="text-white w-3 h-3" />
        <span>{t(I18nKey.COMMON$ERROR)}</span>
      </span>
    );
  }

  return null;
}
