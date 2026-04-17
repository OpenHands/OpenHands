import { useMemo } from "react";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useV1ConversationStateStore } from "#/stores/v1-conversation-state-store";
import { AgentState } from "#/types/agent-state";
import { V1ExecutionStatus } from "#/types/v1/core/base/common";

/**
 * Maps V1 agent status to V0 AgentState
 */
function mapV1StatusToV0State(status: V1ExecutionStatus | null): AgentState {
  if (!status) {
    return AgentState.LOADING;
  }

  switch (status) {
    case V1ExecutionStatus.IDLE:
      return AgentState.AWAITING_USER_INPUT;
    case V1ExecutionStatus.RUNNING:
      return AgentState.RUNNING;
    case V1ExecutionStatus.PAUSED:
      return AgentState.PAUSED;
    case V1ExecutionStatus.WAITING_FOR_CONFIRMATION:
      return AgentState.AWAITING_USER_CONFIRMATION;
    case V1ExecutionStatus.FINISHED:
      return AgentState.FINISHED;
    case V1ExecutionStatus.ERROR:
      return AgentState.ERROR;
    case V1ExecutionStatus.STUCK:
      return AgentState.ERROR; // Map STUCK to ERROR for now
    default:
      return AgentState.LOADING;
  }
}

export interface UseAgentStateResult {
  curAgentState: AgentState;
  executionStatus?: V1ExecutionStatus | null;
}

/**
 * Unified hook that returns the current agent state
 * - For V0 conversations: Returns state from useAgentStore
 * - For V1 conversations: Returns mapped state from useV1ConversationStateStore
 */
export function useAgentState(): UseAgentStateResult {
  const liveExecutionStatus = useV1ConversationStateStore(
    (state) => state.execution_status,
  );
  const fallbackExecutionStatus =
    useActiveConversation().data?.execution_status ?? null;

  const executionStatus = liveExecutionStatus ?? fallbackExecutionStatus;
  const curAgentState = useMemo(
    () => mapV1StatusToV0State(executionStatus),
    [executionStatus],
  );

  return { curAgentState, executionStatus };
}
