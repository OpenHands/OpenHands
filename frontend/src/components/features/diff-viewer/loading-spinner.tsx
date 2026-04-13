import { Spinner } from "#/components/shared/spinner";
import { cn } from "#/utils/utils";

export interface LoadingSpinnerProps {
  className?: string;
}

export function LoadingSpinner({ className }: LoadingSpinnerProps) {
  return (
    <div className="flex items-center justify-center">
      <Spinner
        size="md"
        className={cn("border-4 border-gray-200 border-t-blue-500", className)}
      />
    </div>
  );
}
