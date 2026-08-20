import { useAgentState } from "#/hooks/use-agent-state";
import {
  RUNTIME_INACTIVE_STATES,
  RUNTIME_STARTING_STATES,
} from "#/types/agent-state";
import { useActiveConversation } from "./query/use-active-conversation";
import { isExecutionActive } from "#/utils/status";

interface UseRuntimeIsReadyOptions {
  allowAgentError?: boolean;
}

export const useRuntimeIsReady = ({
  allowAgentError = false,
}: UseRuntimeIsReadyOptions = {}): boolean => {
  const { data: conversation } = useActiveConversation();
  const { curAgentState } = useAgentState();
  const inactiveStates = allowAgentError
    ? RUNTIME_STARTING_STATES
    : RUNTIME_INACTIVE_STATES;

  return (
    isExecutionActive(conversation?.execution_status) &&
    !inactiveStates.includes(curAgentState)
  );
};
