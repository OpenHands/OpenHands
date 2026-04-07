import { cn } from "#/utils/utils";

type SpinnerSize = "sm" | "md" | "lg" | "xl";

interface SpinnerProps {
  size?: SpinnerSize;
  label?: string;
  className?: string;
  labelClassName?: string;
}

const SIZE_CLASSES: Record<SpinnerSize, string> = {
  sm: "h-4 w-4 border-2",
  md: "h-6 w-6 border-2",
  lg: "h-8 w-8 border-3",
  xl: "h-16 w-16 border-4",
};

export function Spinner({
  size = "md",
  label,
  className,
  labelClassName,
}: SpinnerProps) {
  const spinner = (
    <div
      data-testid="spinner"
      role="status"
      aria-label={label ?? "Loading"}
      className={cn(
        "animate-spin rounded-full border-current border-t-transparent",
        SIZE_CLASSES[size],
        className,
      )}
    />
  );

  if (label) {
    return (
      <div className="flex flex-col items-center gap-2">
        {spinner}
        <span className={cn("text-sm", labelClassName)}>{label}</span>
      </div>
    );
  }

  return spinner;
}
