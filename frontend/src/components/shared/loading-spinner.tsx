import { cn } from "#/utils/utils";
import { Spinner } from "./spinner";

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
  return (
    <Spinner
      testId="loading-spinner"
      size={size === "small" ? "md" : "lg"}
      spinnerClassName={cn(className, innerClassName, outerClassName)}
    />
  );
}
