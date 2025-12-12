import PaperclipIcon from "#/icons/paper-clip.svg?react";
import { cn } from "#/utils/utils";

export interface ChatAddFileButtonProps {
  handleFileIconClick: () => void;
  disabled?: boolean;
}

export function ChatAddFileButton({
  handleFileIconClick,
  disabled = false,
}: ChatAddFileButtonProps) {
  return (
    <button
      type="button"
      className={cn(
        "h-[25px] relative shrink-0 w-[13px] cursor-pointer transition-all duration-200 hover:scale-110 active:scale-95",
        disabled && "cursor-not-allowed",
      )}
      data-name="Shape"
      data-testid="paperclip-icon"
      onClick={handleFileIconClick}
    >
      <PaperclipIcon
        className="block max-w-none w-[13px] h-[25px]"
        color={disabled ? "#959CB2" : "white"}
      />
    </button>
  );
}
