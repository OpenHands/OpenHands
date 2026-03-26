import { LoaderCircle } from "lucide-react";
import { cn } from "#/utils/utils";

interface LoadingSpinnerProps {
  size?: "small" | "large";
  className?: string;
  "aria-label"?: string;
}

const sizeClasses = {
  small: "w-[25px] h-[25px]",
  large: "w-[50px] h-[50px]",
};

export function LoadingSpinner({
  size = "small",
  className,
  "aria-label": ariaLabel = "Loading",
}: LoadingSpinnerProps) {
  return (
    <LoaderCircle
      data-testid="loading-spinner"
      role="status"
      aria-label={ariaLabel}
      className={cn("animate-spin", sizeClasses[size], className)}
    />
  );
}
