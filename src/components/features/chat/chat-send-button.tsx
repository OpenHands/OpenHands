import { ArrowUp, Loader2 } from "lucide-react";
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
  return (
    <button
      type="button"
      className={cn(
        "flex items-center justify-center rounded-full border border-white size-8",
        disabled
          ? "cursor-not-allowed border-[var(--oh-muted)]"
          : "cursor-pointer hover:bg-white/10",
        buttonClassName,
      )}
      data-name="arrow-up-circle-fill"
      data-testid="submit-button"
      onClick={handleSubmit}
      disabled={disabled}
      aria-busy={isPending}
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
