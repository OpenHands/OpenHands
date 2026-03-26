import { cn } from "#/utils/utils";
import { LoadingSpinner } from "#/components/shared/loading-spinner";

interface LoaderProps {
  size?: "small" | "medium" | "large";
  className?: string;
}

const loaderSizeClasses = {
  small: "w-3 h-3",
  medium: "w-4 h-4",
  large: "w-5 h-5",
};

export function Loader({ size = "medium", className }: LoaderProps) {
  return (
    <div
      data-testid="loader"
      className={cn("flex items-center justify-center", className)}
    >
      <LoadingSpinner className={loaderSizeClasses[size]} />
    </div>
  );
}
