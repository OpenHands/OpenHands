import { cn } from "#/utils/utils";
import { dropdownMenuRowForegroundClassName } from "#/utils/dropdown-classes";

interface ContextMenuListItemProps {
  testId?: string;
  onClick: (event: React.MouseEvent<HTMLButtonElement>) => void;
  isDisabled?: boolean;
  className?: string;
}

export function ContextMenuListItem({
  children,
  testId,
  onClick,
  isDisabled,
  className,
}: React.PropsWithChildren<ContextMenuListItemProps>) {
  return (
    <button
      data-testid={testId || "context-menu-list-item"}
      type="button"
      onClick={onClick}
      disabled={isDisabled}
      className={cn(
        dropdownMenuRowForegroundClassName,
        "text-nowrap",
        className,
      )}
    >
      {children}
    </button>
  );
}
