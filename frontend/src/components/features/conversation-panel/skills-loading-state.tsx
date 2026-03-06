import { Spinner } from "#/components/shared/spinner";

export function SkillsLoadingState() {
  return (
    <div className="flex justify-center items-center py-8">
      <Spinner
        size="lg"
        className="text-primary"
        testId="skills-loading-spinner"
      />
    </div>
  );
}
