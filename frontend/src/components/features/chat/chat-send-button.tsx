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
        "flex items-center justify-center rounded-full border border-content size-[35px]",
        disabled
          ? "cursor-not-allowed border-neutral-600"
          : "cursor-pointer hover:bg-[#959CB2]",
        buttonClassName,
      )}
      data-name="arrow-up-circle-fill"
      data-testid="submit-button"
      onClick={handleSubmit}
      disabled={disabled}
    >
      <ArrowUp
        className="text-content"
        color={disabled ? "#959CB2" : "currentColor"}
      />
    </button>
  );
}
