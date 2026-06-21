import { useMemo } from "react";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useV1ConversationStateStore } from "#/stores/v1-conversation-state-store";
import { AgentState } from "#/types/agent-state";
import { V1ExecutionStatus } from "#/types/v1/core/base/common";
import { V1SandboxStatus } from "#/api/sandbox-service/sandbox-service.types";

/**
 * Maps V1 agent status to V0 AgentState
 */
function mapV1StatusToV0State(
  status: V1ExecutionStatus | null,
  sandboxStatus: V1SandboxStatus | undefined,
): AgentState {
  // For archived conversations (sandbox MISSING), return STOPPED to avoid loading states
  // The conversation is read-only and won't resume, so we should not show "starting" indicators
  if (sandboxStatus === "MISSING") {
    return AgentState.STOPPED;
  }

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
  isArchived: boolean;
}

/**
 * Unified hook that returns the current agent state
 * - For V0 conversations: Returns state from useAgentStore
 * - For V1 conversations: Returns mapped state from useV1ConversationStateStore
 * - For archived conversations (sandbox MISSING): Returns STOPPED state
 */
export function useAgentState(): UseAgentStateResult {
  const liveExecutionStatus = useV1ConversationStateStore(
    (state) => state.execution_status,
  );
  const conversation = useActiveConversation().data;
  const fallbackExecutionStatus = conversation?.execution_status ?? null;
  const sandboxStatus = conversation?.sandbox_status;

  const executionStatus = liveExecutionStatus ?? fallbackExecutionStatus;
  const isArchived = sandboxStatus === "MISSING";

  const curAgentState = useMemo(
    () => mapV1StatusToV0State(executionStatus, sandboxStatus),
    [executionStatus, sandboxStatus],
  );

  return { curAgentState, executionStatus, isArchived };
}
