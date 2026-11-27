import { cn } from "#/utils/utils";
import {
  dropdownMenuRowGapClassName,
  dropdownMenuRowIconWrapperClassName,
} from "#/utils/dropdown-classes";
import CheckIcon from "#/icons/checkmark.svg?react";

interface ContextMenuIconTextProps {
  icon: React.ComponentType<{ className?: string }>;
  text: string;
  className?: string;
  iconClassName?: string;
  isActive?: boolean;
}

export function ContextMenuIconText({
  icon: Icon,
  text,
  className,
  iconClassName,
  isActive = false,
}: ContextMenuIconTextProps) {
  return (
    <div
      className={cn(
        "flex min-w-0 items-center",
        dropdownMenuRowGapClassName,
        className,
      )}
    >
      <Icon
        className={cn(
          "h-4 w-4",
          dropdownMenuRowIconWrapperClassName,
          iconClassName,
        )}
      />
      <span className="min-w-0 flex-1 leading-5">{text}</span>
      {isActive && (
        <CheckIcon width={14} height={14} className="shrink-0" aria-hidden />
      )}
    </div>
  );
}
