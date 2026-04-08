import { useAgentState } from "#/hooks/use-agent-state";
import {
  RUNTIME_INACTIVE_STATES,
  RUNTIME_STARTING_STATES,
} from "#/types/agent-state";
import { useActiveConversation } from "./query/use-active-conversation";

interface UseRuntimeIsReadyOptions {
  allowAgentError?: boolean;
}

/**
 * Hook to determine if the runtime is ready for operations
 *
 * @returns boolean indicating if the runtime is ready
 */
export const useRuntimeIsReady = ({
  allowAgentError = false,
}: UseRuntimeIsReadyOptions = {}): boolean => {
  const { data: conversation } = useActiveConversation();
  const { curAgentState } = useAgentState();
  const inactiveStates = allowAgentError
    ? RUNTIME_STARTING_STATES
    : RUNTIME_INACTIVE_STATES;

  return (
    conversation?.sandbox_status === "RUNNING" &&
    !inactiveStates.includes(curAgentState)
  );
};
