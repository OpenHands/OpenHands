import { LoadingSpinner } from "#/components/shared/loading-spinner";

export function AgentLoading() {
  return (
    <div data-testid="agent-loading-spinner">
      <LoadingSpinner className="w-4 h-4 text-white" />
    </div>
  );
}
