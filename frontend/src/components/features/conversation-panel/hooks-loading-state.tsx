import { Spinner } from "#/components/shared/spinner";

export function HooksLoadingState() {
  return (
    <div className="flex justify-center items-center py-8">
      <Spinner size="lg" />
    </div>
  );
}
