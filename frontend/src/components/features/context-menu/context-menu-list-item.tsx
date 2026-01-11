import { cn } from "#/utils/utils";

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
        "text-sm px-4 h-10 w-full text-start hover:bg-surface-hover cursor-pointer",
        "disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-transparent text-nowrap",
        className,
      )}
    >
      {children}
    </button>
  );
}
