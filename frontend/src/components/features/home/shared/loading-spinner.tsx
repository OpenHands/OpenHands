import { cn } from "#/utils/utils";
import { Spinner } from "#/components/shared/spinner";

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
      data-testid={testId}
      className={cn(
        "absolute top-1/2 transform -translate-y-1/2",
        hasSelection ? "right-11" : "right-6",
      )}
    >
      <Spinner size="sm" className="text-blue-500" />
    </div>
  );
}
