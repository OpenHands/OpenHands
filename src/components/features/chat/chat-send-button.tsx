import { ArrowUp } from "lucide-react";
import { cn } from "#/utils/utils";

export interface ChatSendButtonProps {
  buttonClassName: string;
  handleSubmit: () => void;
  disabled: boolean;
}

export function ChatSendButton({
  buttonClassName,
  handleSubmit,
  disabled,
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
    >
      <ArrowUp
        className="w-4 h-4"
        color={disabled ? "var(--oh-muted)" : "white"}
      />
    </button>
  );
}
