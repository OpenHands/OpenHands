import React from "react";
import { cn } from "#/utils/utils";
import { Spinner } from "#/ui/spinner";

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
      <Spinner size="sm" className="text-blue-500" testId={testId} />
    </div>
  );
}
