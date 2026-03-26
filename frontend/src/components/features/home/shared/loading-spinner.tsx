import { cn } from "#/utils/utils";
import { LoadingSpinner as SharedLoadingSpinner } from "#/components/shared/loading-spinner";

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
      <div data-testid={testId}>
        <SharedLoadingSpinner className="w-4 h-4" />
      </div>
    </div>
  );
}
