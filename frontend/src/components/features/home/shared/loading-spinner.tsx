import React from "react";
import { cn } from "#/utils/utils";
import { Spinner } from "#/components/shared/spinner";

interface LoadingSpinnerProps {
  hasSelection?: boolean;
  testId?: string;
  className?: string;
}

export function LoadingSpinner({
  hasSelection,
  testId = "dropdown-loading",
  className,
}: LoadingSpinnerProps) {
  return (
    <div
      className={cn(
        "absolute top-1/2 transform -translate-y-1/2",
        hasSelection ? "right-11" : "right-6",
        className,
      )}
    >
      <Spinner size="sm" />
      <div data-testid={testId} className="sr-only" />
    </div>
  );
}
