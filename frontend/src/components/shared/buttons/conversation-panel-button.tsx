import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import ListIcon from "#/icons/list.svg?react";
import { TooltipButton } from "./tooltip-button";
import { cn } from "#/utils/utils";

interface ConversationPanelButtonProps {
  isOpen: boolean;
  onClick: () => void;
  disabled?: boolean;
}

export function ConversationPanelButton({
  isOpen,
  onClick,
  disabled = false,
}: ConversationPanelButtonProps) {
  const { t } = useTranslation();

  return (
    <TooltipButton
      testId="toggle-conversation-panel"
      tooltip={t(I18nKey.SIDEBAR$CONVERSATIONS)}
      ariaLabel={t(I18nKey.SIDEBAR$CONVERSATIONS)}
      onClick={onClick}
      disabled={disabled}
    >
      <ListIcon
        width={24}
        height={24}
        className={cn(
          "cursor-pointer",
          isOpen ? "text-white" : "text-[#B1B9D3]",
          disabled && "opacity-50",
        )}
      />
    </TooltipButton>
  );
}
