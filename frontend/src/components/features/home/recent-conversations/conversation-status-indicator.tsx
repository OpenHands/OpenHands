import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { ConversationStatus } from "#/types/conversation-status";
import { cn, getConversationStatusLabel } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";

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
        return "bg-[#3C3C49]";
      case "RUNNING":
        return "bg-[#1FBD53]";
      case "STARTING":
        return "bg-[#FFD43B]";
      case "ERROR":
        return "bg-[#FF684E]";
      default:
        return "bg-[#3C3C49]";
    }
  }, [conversationStatus]);

  const statusLabel = t(
    getConversationStatusLabel(conversationStatus) as I18nKey,
  );

  return (
    <StyledTooltip
      content={statusLabel}
      placement="right"
      showArrow
      tooltipClassName="bg-[#1a1a1a] text-white text-xs shadow-lg"
    >
      <div
        className={cn(
          "w-1.5 h-1.5 rounded-full",
          conversationStatusBackgroundColor,
        )}
      />
    </StyledTooltip>
  );
}
