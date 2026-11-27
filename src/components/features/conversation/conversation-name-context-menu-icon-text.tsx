import { cn } from "#/utils/utils";
import {
  dropdownMenuRowGapClassName,
  dropdownMenuRowIconWrapperClassName,
} from "#/utils/dropdown-classes";

interface ConversationNameContextMenuIconTextProps {
  icon: React.ReactNode;
  text: string;
  className?: string;
}

export function ConversationNameContextMenuIconText({
  icon,
  text,
  className,
}: ConversationNameContextMenuIconTextProps) {
  return (
    <div
      className={cn(
        "flex min-w-0 w-full items-center",
        dropdownMenuRowGapClassName,
        className,
      )}
    >
      <span className={dropdownMenuRowIconWrapperClassName} aria-hidden>
        {icon}
      </span>
      <span className="min-w-0 flex-1 truncate">{text}</span>
    </div>
  );
}
