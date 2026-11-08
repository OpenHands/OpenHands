import { useMemo } from "react";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useConversationStateStore } from "#/stores/conversation-state-store";
import { useConversationStore } from "#/stores/conversation-store";
import { AgentState } from "#/types/agent-state";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

/**
 * Maps agent execution status to AgentState
 */
function mapExecutionStatusToAgentState(
  status: ExecutionStatus | null,
): AgentState {
  if (!status) {
    return AgentState.LOADING;
  }

  switch (status) {
    case ExecutionStatus.IDLE:
      return AgentState.AWAITING_USER_INPUT;
    case ExecutionStatus.RUNNING:
      return AgentState.RUNNING;
    case ExecutionStatus.PAUSED:
      return AgentState.PAUSED;
    case ExecutionStatus.WAITING_FOR_CONFIRMATION:
      return AgentState.AWAITING_USER_CONFIRMATION;
    case ExecutionStatus.FINISHED:
      return AgentState.FINISHED;
    case ExecutionStatus.ERROR:
      return AgentState.ERROR;
    case ExecutionStatus.STUCK:
      return AgentState.ERROR; // Map STUCK to ERROR for now
    default:
      return AgentState.LOADING;
  }
}

export interface UseAgentStateResult {
  curAgentState: AgentState;
  executionStatus?: ExecutionStatus | null;
}

/**
 * Returns the current agent state from conversation execution status.
 *
 * Defaults to the conversation in the current route. Pass `conversationId`
 * to read another conversation's status instead — e.g. a local planner
 * helper conversation, whose run/idle transitions are tracked separately
 * from the main conversation's (see conversation-state-store.ts).
 */
export function useAgentState(conversationId?: string): UseAgentStateResult {
  const { conversationId: routeConversationId } = useOptionalConversationId();
  const targetConversationId = conversationId ?? routeConversationId;
  const isRouteConversation =
    !conversationId || conversationId === routeConversationId;

  const liveExecutionStatus = useConversationStateStore((state) =>
    targetConversationId
      ? (state.executionStatusByConversation[targetConversationId] ?? null)
      : null,
  );
  // The REST fallback only ever describes the route's own conversation, so it
  // only applies when no other conversation was explicitly requested.
  const routeConversationExecutionStatus =
    useActiveConversation().data?.execution_status ?? null;
  const fallbackExecutionStatus = isRouteConversation
    ? routeConversationExecutionStatus
    : null;

  const executionStatus = liveExecutionStatus ?? fallbackExecutionStatus;
  const curAgentState = useMemo(
    () => mapExecutionStatusToAgentState(executionStatus),
    [executionStatus],
  );

  return { curAgentState, executionStatus };
}

export interface UsePlanningAgentStateResult {
  localPlanningConversationId: string | null;
  curPlanningAgentState: AgentState;
  /** Running or loading. `false` (not "unknown") when there's no planner yet. */
  isPlanningAgentRunning: boolean;
}

/**
 * The local planner helper's own state — read separately from the main
 * conversation's via `useAgentState` since the two run independently.
 */
export function usePlanningAgentState(): UsePlanningAgentStateResult {
  const localPlanningConversationId = useConversationStore(
    (state) => state.localPlanningConversationId,
  );
  const { curAgentState: curPlanningAgentState } = useAgentState(
    localPlanningConversationId ?? undefined,
  );
  const isPlanningAgentRunning =
    !!localPlanningConversationId &&
    (curPlanningAgentState === AgentState.RUNNING ||
      curPlanningAgentState === AgentState.LOADING);
  return {
    localPlanningConversationId,
    curPlanningAgentState,
    isPlanningAgentRunning,
  };
}
