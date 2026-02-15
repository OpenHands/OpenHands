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
        className="p-1.5 bg-transparent border-0 rounded-lg transition-all duration-150 hover:bg-[#18181B] active:scale-95"
      >
        <ListIcon
          width={22}
          height={22}
          className={cn(
            "cursor-pointer transition-colors",
            isOpen ? "text-white" : "text-[#A1A1AA]",
            disabled && "opacity-50",
          )}
        />
      </button>
    </StyledTooltip>
  );
}
