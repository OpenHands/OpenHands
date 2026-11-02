import { ArrowUp, Loader2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

export interface ChatSendButtonProps {
  buttonClassName: string;
  handleSubmit: () => void;
  disabled: boolean;
  isPending?: boolean;
}

export function ChatSendButton({
  buttonClassName,
  handleSubmit,
  disabled,
  isPending = false,
}: ChatSendButtonProps) {
  const { t } = useTranslation("openhands");

  return (
    <button
      type="button"
      className={cn(
        "flex items-center justify-center rounded-full border border-contrast size-8",
        disabled
          ? "cursor-not-allowed border-muted"
          : "cursor-pointer hover:bg-contrast/10",
        buttonClassName,
      )}
      data-name="arrow-up-circle-fill"
      data-testid="submit-button"
      onClick={handleSubmit}
      disabled={disabled}
      aria-busy={isPending}
      aria-label={t(I18nKey.CHAT_INTERFACE$TOOLTIP_SEND_MESSAGE)}
    >
      {isPending ? (
        <Loader2
          data-testid="submit-button-pending-icon"
          className="size-4 animate-spin text-[var(--oh-muted)]"
          aria-hidden
        />
      ) : (
        <ArrowUp
          className="w-4 h-4"
          color={disabled ? "var(--oh-muted)" : "white"}
        />
      )}
    </button>
  );
}
