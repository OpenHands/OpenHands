import { Spinner } from "#/components/shared/spinner";

export function AgentLoading() {
  return (
    <div data-testid="agent-loading-spinner">
      <Spinner size="sm" className="text-white" testId="agent-loading-icon" />
    </div>
  );
}
