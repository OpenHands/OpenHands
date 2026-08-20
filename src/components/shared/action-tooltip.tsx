import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { StyledTooltip } from "./buttons/styled-tooltip";

interface ActionTooltipProps {
  type: "confirm" | "reject";
  onClick: () => void;
}

export function ActionTooltip({ type, onClick }: ActionTooltipProps) {
  const { t } = useTranslation("openhands");

  const isConfirm = type === "confirm";

  const ariaLabel = isConfirm
    ? t(I18nKey.ACTION$CONFIRM)
    : t(I18nKey.ACTION$REJECT);

  const content = isConfirm
    ? t(I18nKey.CHAT_INTERFACE$USER_CONFIRMED)
    : t(I18nKey.CHAT_INTERFACE$USER_REJECTED);

  const buttonLabel = isConfirm
    ? `${t(I18nKey.CHAT_INTERFACE$INPUT_CONTINUE_MESSAGE)} ⌘↩`
    : `${t(I18nKey.BUTTON$CANCEL)} ⇧⌘⌫`;

  return (
    <StyledTooltip closeDelay={100} content={content}>
      <button
        data-testid={`action-${type}-button`}
        type="button"
        aria-label={ariaLabel}
        className={cn(
          "rounded px-2 h-6.5 text-sm font-normal leading-5 cursor-pointer hover:opacity-80",
          type === "confirm" ? "bg-tertiary text-white" : "bg-white text-base",
        )}
        onClick={onClick}
      >
        {buttonLabel}
      </button>
    </StyledTooltip>
  );
}
