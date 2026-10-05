import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { isApplePlatform } from "#/utils/is-apple-platform";
import { StyledTooltip } from "./buttons/styled-tooltip";

const SHORTCUT_HINTS = {
  apple: { confirm: "⌘↩", reject: "⇧⌘⌫" },
  other: { confirm: "Ctrl+↩", reject: "Ctrl+⇧+⌫" },
} as const;

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

  const shortcutHint = SHORTCUT_HINTS[isApplePlatform() ? "apple" : "other"];

  const buttonLabel = isConfirm
    ? `${t(I18nKey.CHAT_INTERFACE$INPUT_CONTINUE_MESSAGE)} ${shortcutHint.confirm}`
    : `${t(I18nKey.BUTTON$CANCEL)} ${shortcutHint.reject}`;

  return (
    <StyledTooltip closeDelay={100} content={content}>
      <button
        data-testid={`action-${type}-button`}
        type="button"
        aria-label={ariaLabel}
        className={cn(
          "rounded px-2 h-6.5 text-sm font-normal leading-5 cursor-pointer hover:opacity-80",
          type === "confirm"
            ? "bg-tertiary text-contrast"
            : "bg-contrast text-base",
        )}
        onClick={onClick}
      >
        {buttonLabel}
      </button>
    </StyledTooltip>
  );
}
