import { LoadingSpinner as SharedLoadingSpinner } from "#/components/shared/loading-spinner";

export function LoadingSpinner() {
  return (
    <div data-testid="dropdown-loading">
      <SharedLoadingSpinner className="w-4 h-4" />
    </div>
  );
}
