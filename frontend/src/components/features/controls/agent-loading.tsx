import { Spinner } from "#/ui/spinner";

export function AgentLoading() {
  return (
    <div data-testid="agent-loading-spinner">
      <Spinner size="sm" className="text-white" />
    </div>
  );
}
