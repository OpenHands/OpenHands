import { LoaderCircle } from "lucide-react";
import { cn } from "#/utils/utils";

type SpinnerSize = "sm" | "md" | "lg" | "xl";

interface SpinnerProps {
  size?: SpinnerSize;
  label?: string;
  className?: string;
  labelClassName?: string;
  wrapperClassName?: string;
  testId?: string;
}

const SPINNER_SIZE_CLASS: Record<SpinnerSize, string> = {
  sm: "h-4 w-4",
  md: "h-6 w-6",
  lg: "h-8 w-8",
  xl: "h-16 w-16",
};

const LABEL_SIZE_CLASS: Record<SpinnerSize, string> = {
  sm: "text-xs",
  md: "text-sm",
  lg: "text-base",
  xl: "text-2xl",
};

export function Spinner({
  size = "md",
  label,
  className,
  labelClassName,
  wrapperClassName,
  testId = "spinner",
}: SpinnerProps) {
  return (
    <div
      className={cn("flex items-center justify-center gap-2", wrapperClassName)}
    >
      <LoaderCircle
        data-testid={testId}
        className={cn(
          "animate-spin shrink-0",
          SPINNER_SIZE_CLASS[size],
          className,
        )}
        aria-hidden={Boolean(label)}
      />
      {label ? (
        <span
          className={cn("leading-5", LABEL_SIZE_CLASS[size], labelClassName)}
        >
          {label}
        </span>
      ) : null}
    </div>
  );
}
