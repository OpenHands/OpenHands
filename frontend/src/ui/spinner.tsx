import React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "#/utils/utils";
import { Typography } from "#/ui/typography";

const spinnerVariants = cva(
  "animate-spin rounded-full border-current border-t-transparent",
  {
    variants: {
      size: {
        sm: "w-4 h-4 border-2 border-t-2",
        md: "w-5 h-5 border-2 border-t-2",
        lg: "w-8 h-8 border-2 border-t-2",
        xl: "w-16 h-16 border-4 border-t-4",
      },
    },
    defaultVariants: {
      size: "md",
    },
  },
);

export interface SpinnerProps extends VariantProps<typeof spinnerVariants> {
  className?: string;
  spinnerClassName?: string;
  label?: React.ReactNode;
  labelClassName?: string;
  testId?: string;
}

export function Spinner({
  size,
  className,
  spinnerClassName,
  label,
  labelClassName,
  testId,
}: SpinnerProps) {
  return (
    <div
      data-testid={testId}
      role="status"
      aria-live="polite"
      className={cn("inline-flex items-center gap-2", className)}
    >
      <div className={cn(spinnerVariants({ size }), spinnerClassName)} />
      {label ? (
        <Typography.Text className={cn("text-sm", labelClassName)}>
          {label}
        </Typography.Text>
      ) : null}
    </div>
  );
}
