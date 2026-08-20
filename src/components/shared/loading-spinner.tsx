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
  const sizeStyle =
    size === "small" ? "w-[25px] h-[25px]" : "w-[50px] h-[50px]";

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
