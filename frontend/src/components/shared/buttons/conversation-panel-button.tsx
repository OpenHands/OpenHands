import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import ListIcon from "#/icons/list.svg?react";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
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

  const label = t(I18nKey.SIDEBAR$CONVERSATIONS);

  return (
    <StyledTooltip content={label}>
      <button
        type="button"
        data-testid="toggle-conversation-panel"
        aria-label={label}
        onClick={onClick}
        disabled={disabled}
        className="p-0 bg-transparent border-0"
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
      </button>
    </StyledTooltip>
  );
}
