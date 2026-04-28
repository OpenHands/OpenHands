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
        "px-4 py-2 font-medium border-b-2 transition-colors",
        isActive
          ? "border-primary text-gray-100"
          : "border-transparent hover:text-gray-700 dark:hover:text-gray-300",
        disabled && "opacity-50 cursor-not-allowed",
        className,
      )}
      onClick={onClick}
    >
      {children}
    </button>
  );
}
