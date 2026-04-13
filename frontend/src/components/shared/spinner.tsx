import { cn } from "#/utils/utils";

export type SpinnerSize = "xs" | "sm" | "md" | "lg" | "xl";

interface SpinnerProps {
  size?: SpinnerSize;
  label?: string;
  className?: string;
  testId?: string;
}

const sizeClasses: Record<SpinnerSize, string> = {
  xs: "h-3 w-3",
  sm: "h-4 w-4",
  md: "h-6 w-6",
  lg: "h-8 w-8",
  xl: "h-16 w-16",
};

export function Spinner({
  size = "md",
  label,
  className,
  testId,
}: SpinnerProps) {
  return (
    <div
      data-testid={testId}
      className={cn("flex items-center justify-center", className)}
    >
      <div
        className={cn(
          "animate-spin rounded-full border-2 border-t-transparent border-primary",
          sizeClasses[size],
        )}
        role="status"
        aria-label={label ?? "Loading"}
      />
      {label && <span className="ml-2">{label}</span>}
    </div>
  );
}
