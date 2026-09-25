import LoadingSpinnerOuter from "#/icons/loading-outer.svg?react";
import { cn } from "#/utils/utils";

interface LoadingSpinnerProps {
  size: "small" | "large";
  className?: string;
  outerClassName?: string;
}

export function LoadingSpinner({
  size,
  className,
  outerClassName,
}: LoadingSpinnerProps) {
  const sizeStyle = size === "small" ? "w-6.25 h-6.25" : "w-12.5 h-12.5";

  return (
    <div
      data-testid="loading-spinner"
      className={cn("relative", sizeStyle, className)}
    >
      <LoadingSpinnerOuter
        className={cn("absolute animate-spin", sizeStyle, outerClassName)}
      />
    </div>
  );
}
