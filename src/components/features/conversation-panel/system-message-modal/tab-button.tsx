import { cn } from "#/utils/utils";

interface TabButtonProps {
  isActive: boolean;
  children: React.ReactNode;
  onClick: () => void;
  className?: string;
  disabled?: boolean;
}

export function TabButton({
  isActive,
  children,
  onClick,
  className,
  disabled = false,
}: TabButtonProps) {
  return (
    <button
      type="button"
      disabled={disabled}
      className={cn(
        "px-4 py-2 font-normal border-b-2 transition-colors",
        isActive
          ? "border-foreground text-foreground"
          : "border-transparent text-[var(--oh-muted)] hover:text-[var(--oh-foreground)]",
        disabled && "opacity-50 cursor-not-allowed",
        className,
      )}
      onClick={onClick}
      aria-selected={isActive}
      role="tab"
    >
      {children}
    </button>
  );
}
