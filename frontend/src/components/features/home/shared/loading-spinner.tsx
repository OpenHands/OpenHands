import React from "react";
import { Spinner } from "#/components/shared/spinner";
import { cn } from "#/utils/utils";

interface LoadingSpinnerProps {
  hasSelection: boolean;
  testId?: string;
}

export function LoadingSpinner({
  hasSelection,
  testId = "dropdown-loading",
}: LoadingSpinnerProps) {
  return (
    <div
      className={cn(
        "absolute top-1/2 transform -translate-y-1/2",
        hasSelection ? "right-11" : "right-6",
      )}
    >
      <Spinner size="sm" testId={testId} />
    </div>
  );
}
