import LoadingSpinnerOuter from "#/icons/loading-outer.svg?react";
import { cn } from "#/utils/utils";

interface LoadingSpinnerProps {
  size: "small" | "large";
  className?: string;
  innerClassName?: string;
  outerClassName?: string;
}

export function LoadingSpinner({
  size,
  className,
  innerClassName,
  outerClassName,
}: LoadingSpinnerProps) {
  const sizeStyle =
    size === "small" ? "w-[25px] h-[25px]" : "w-[50px] h-[50px]";

  return (
    <div
      data-testid="loading-spinner"
      className={cn("relative", sizeStyle, className)}
    >
      <div
        className={cn(
          "rounded-full border-4 border-[#525252] absolute",
          sizeStyle,
          innerClassName,
        )}
      />
      <LoadingSpinnerOuter
        className={cn("absolute animate-spin", sizeStyle, outerClassName)}
      />
    </div>
  );
}
