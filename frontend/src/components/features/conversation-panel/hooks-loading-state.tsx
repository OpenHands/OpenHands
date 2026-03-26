import { LoadingSpinner } from "#/components/shared/loading-spinner";

export function HooksLoadingState() {
  return (
    <div className="flex justify-center items-center py-8">
      <LoadingSpinner className="h-8 w-8" />
    </div>
  );
}
