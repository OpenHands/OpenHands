import { Spinner, SpinnerSize } from "#/components/shared/spinner";
import { cn } from "#/utils/utils";

interface LoadingSpinnerProps {
  size: "small" | "large";
  className?: string;
}

const sizeMap: Record<"small" | "large", SpinnerSize> = {
  small: "sm",
  large: "lg",
};

export function LoadingSpinner({ size, className }: LoadingSpinnerProps) {
  return (
    <Spinner
      size={sizeMap[size]}
      testId="loading-spinner"
      className={cn("relative", className)}
    />
  );
}
