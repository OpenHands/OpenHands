import { RUNTIME_INACTIVE_STATES } from "#/types/agent-state";
import { useActiveConversation } from "./query/use-active-conversation";
import { useAgentState } from "#/hooks/use-agent-state";

/**
 * Hook to determine if the runtime is ready for operations
 *
 * @returns boolean indicating if the runtime is ready
 */
export const useRuntimeIsReady = (): boolean => {
  const { data: conversation } = useActiveConversation();
  const { curAgentState } = useAgentState();

  return (
    conversation?.status === "RUNNING" &&
    !RUNTIME_INACTIVE_STATES.includes(curAgentState)
  );
};
