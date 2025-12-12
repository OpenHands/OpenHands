import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { ConversationStatus } from "#/types/conversation-status";
import { cn, getConversationStatusLabel } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";
import { TooltipButton } from "#/components/shared/buttons/tooltip-button";

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
    <TooltipButton
      tooltip={statusLabel}
      ariaLabel={statusLabel}
      placement="right"
      showArrow
      className="p-0 border-0 bg-transparent hover:opacity-100"
      tooltipClassName="bg-[#1a1a1a] text-white text-xs shadow-lg"
    >
      <div
        className={cn(
          "w-1.5 h-1.5 rounded-full",
          conversationStatusBackgroundColor,
        )}
      />
    </TooltipButton>
  );
}
