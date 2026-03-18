import { cn } from "#/utils/utils";

interface InteractiveChipProps {
  onClick: () => void;
  testId?: string;
  className?: string;
}

/**
 * Small clickable chip component for actions like "Add"
 * Uses gray background with black text
 */
export function InteractiveChip({
  children,
  onClick,
  testId,
  className,
}: React.PropsWithChildren<InteractiveChipProps>) {
  return (
    <button
      type="button"
      data-testid={testId}
      onClick={onClick}
      className={cn(
        "bg-[#E4E4E4] px-2 rounded-[100px] text-black text-sm text-center font-semibold cursor-pointer",
        "hover:bg-[#D4D4D4] transition-colors",
        className,
      )}
    >
      {children}
    </button>
  );
}
