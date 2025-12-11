import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { Tooltip } from "@heroui/react";
import { ConversationStatus } from "#/types/conversation-status";
import { cn, getConversationStatusLabel } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";

interface ConversationStatusIndicatorProps {
  conversationStatus: ConversationStatus;
}

export function ConversationStatusIndicator({
  conversationStatus,
}: ConversationStatusIndicatorProps) {
  const { t } = useTranslation();

  const conversationStatusBackgroundColor = useMemo(() => {
    switch (conversationStatus) {
      case "STOPPED":
        return "bg-[#3C3C49]"; // Inactive/stopped - grey
      case "RUNNING":
        return "bg-[#1FBD53]"; // Running/online - green
      case "STARTING":
        return "bg-[#FFD43B]"; // Busy/starting - yellow
      case "ERROR":
        return "bg-[#FF684E]"; // Error - red
      default:
        return "bg-[#3C3C49]"; // Default to grey for unknown states
    }
  }, [conversationStatus]);

  const statusLabel = t(
    getConversationStatusLabel(conversationStatus) as I18nKey,
  );

  return (
    <Tooltip
      content={statusLabel}
      closeDelay={100}
      placement="right"
      showArrow
      className="bg-[#1a1a1a] text-white text-xs shadow-lg"
    >
      <span
        aria-label={statusLabel}
        className="p-0 border-0 bg-transparent hover:opacity-100 inline-flex"
      >
        <span
          className={cn(
            "w-1.5 h-1.5 rounded-full",
            conversationStatusBackgroundColor,
          )}
        />
      </span>
    </Tooltip>
  );
}
