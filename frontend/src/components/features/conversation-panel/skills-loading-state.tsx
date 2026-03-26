import { LoadingSpinner } from "#/components/shared/loading-spinner";

export function SkillsLoadingState() {
  return (
    <div className="flex justify-center items-center py-8">
      <LoadingSpinner size="small" className="h-8 w-8" />
    </div>
  );
}
