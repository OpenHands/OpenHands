import { LoaderCircle } from "lucide-react";

export function AgentLoading() {
  return (
    <div data-testid="agent-loading-spinner">
      <LoaderCircle className="animate-spin w-4 h-4 text-current" />
    </div>
  );
}
