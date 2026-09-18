import { cn } from "#/utils/utils";
import CloseIcon from "#/icons/close.svg?react";

interface RemoveButtonProps {
  onClick: () => void;
  className?: React.HTMLAttributes<HTMLDivElement>["className"];
  "aria-label"?: string;
}

export function RemoveButton({
  onClick,
  className,
  "aria-label": ariaLabel,
}: RemoveButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={ariaLabel}
      className={cn(
        "bg-[var(--oh-muted)] rounded-full w-5 h-5 flex items-center justify-center cursor-pointer",
        className,
      )}
    >
      <CloseIcon width={18} height={18} />
    </button>
  );
}
