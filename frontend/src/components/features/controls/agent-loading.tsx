import { Spinner } from "#/components/shared/spinner";

export function AgentLoading() {
  return (
    <Spinner size="sm" className="text-white" testId="agent-loading-spinner" />
  );
}
