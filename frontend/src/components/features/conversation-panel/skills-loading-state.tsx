import { Spinner } from "#/components/shared/spinner";

export function SkillsLoadingState() {
  return (
    <div className="flex justify-center items-center py-8">
      <Spinner size="lg" />
    </div>
  );
}
