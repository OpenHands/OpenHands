import PlayIcon from "#/icons/play-solid.svg?react";
import { cn } from "#/utils/utils";

export interface ChatResumeAgentButtonProps {
  onAgentResumed: () => void;
  disabled?: boolean;
}

export function ChatResumeAgentButton({
  onAgentResumed,
  disabled = false,
}: ChatResumeAgentButtonProps) {
  return (
    <button
      type="button"
      onClick={onAgentResumed}
      data-testid="play-button"
      disabled={disabled}
      className={cn("cursor-pointer", disabled && "cursor-not-allowed")}
    >
      <PlayIcon className="block max-w-none w-4 h-4 text-current" />
    </button>
  );
}
