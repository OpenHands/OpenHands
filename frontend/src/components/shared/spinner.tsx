import React from "react";
import { cn } from "#/utils/utils";

type SpinnerSize = "sm" | "md" | "lg" | "xl";

const SIZE_CLASSES: Record<
  SpinnerSize,
  { wrapper: string; ring: string; border: string }
> = {
  sm: { wrapper: "w-4 h-4", ring: "border-2", border: "border-t-2" },
  md: { wrapper: "w-5 h-5", ring: "border-2", border: "border-t-2" },
  lg: { wrapper: "w-8 h-8", ring: "border-2", border: "border-t-2" },
  xl: { wrapper: "w-16 h-16", ring: "border-4", border: "border-t-4" },
};

export interface SpinnerProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: SpinnerSize;
  label?: React.ReactNode;
  labelClassName?: string;
  spinnerClassName?: string;
}

export function Spinner({
  size = "md",
  label,
  className,
  labelClassName,
  spinnerClassName,
  ...props
}: SpinnerProps) {
  const sizeClasses = SIZE_CLASSES[size];

  return (
    <div
      role="status"
      aria-live="polite"
      className={cn("inline-flex items-center gap-2", className)}
      // eslint-disable-next-line react/jsx-props-no-spreading
      {...props}
    >
      <div
        className={cn(
          "animate-spin rounded-full border-current border-t-transparent",
          sizeClasses.wrapper,
          sizeClasses.ring,
          sizeClasses.border,
          spinnerClassName,
        )}
      />
      {label ? (
        <span className={cn("text-sm", labelClassName)}>{label}</span>
      ) : null}
    </div>
  );
}
