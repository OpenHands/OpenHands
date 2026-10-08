import { useCallback } from "react";
import { useConversationStore } from "#/stores/conversation-store";
import { useHandlePlanClick } from "#/hooks/use-handle-plan-click";
import {
  useMainWebSocketStatus,
  useUnifiedWebSocketStatus,
} from "#/hooks/use-unified-websocket-status";
import { usePlanningAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";
import { PLAN_COMMAND, CODE_COMMAND } from "#/utils/constants";

const PLAN_PREFIX = `${PLAN_COMMAND} `;
const CODE_PREFIX = `${CODE_COMMAND} `;

/**
 * Intercepts "/plan [task]" and "/code [task]" and toggles the conversation's
 * mode like the Code/Plan button, instead of sending them as a chat message.
 * A bare command only switches mode; "<task>" is sent immediately (mode is
 * set synchronously first, so the send routes to the new mode — see
 * conversation-websocket-context.tsx's `getState().conversationMode` read),
 * or becomes the new planner's `initial_message` if it doesn't exist yet.
 *
 * Swallows the command while the relevant agent is running, a planner is
 * being created, or the socket it needs is disconnected — "/code" gates on
 * the main socket alone (not the main+planning merged status) so a momentary
 * planning reconnect can't swallow it, and "/plan <task>" also checks the
 * planner's own running state so it can't be routed into one still mid-run.
 */
export const usePlanModeInterceptor = (
  conversationId: string | null | undefined,
  curAgentState: AgentState,
  onSubmit: (message: string) => void,
) => {
  const setConversationMode = useConversationStore(
    (s) => s.setConversationMode,
  );
  const { handlePlanClick, hasPlanner, isCreatingConversation } =
    useHandlePlanClick();
  const isMainWebSocketConnected = useMainWebSocketStatus() === "OPEN";
  const isWebSocketConnected = useUnifiedWebSocketStatus() === "OPEN";
  const { isPlanningAgentRunning } = usePlanningAgentState();

  return useCallback(
    (message: string) => {
      const trimmed = message.trim();
      const isPlan =
        trimmed === PLAN_COMMAND || trimmed.startsWith(PLAN_PREFIX);
      const isCode =
        trimmed === CODE_COMMAND || trimmed.startsWith(CODE_PREFIX);
      if (!conversationId || (!isPlan && !isCode)) {
        onSubmit(message);
        return;
      }

      if (curAgentState === AgentState.RUNNING || isCreatingConversation) {
        return;
      }

      if (isPlan) {
        if (isPlanningAgentRunning || !isWebSocketConnected) {
          return;
        }
        const task = trimmed.slice(PLAN_COMMAND.length).trim();
        if (task && hasPlanner) {
          // Planner already exists: switch mode, then send normally — the
          // send path reads the just-set mode synchronously, so this routes
          // to the planner rather than the code agent.
          setConversationMode("plan");
          onSubmit(task);
        } else {
          handlePlanClick(undefined, task || undefined);
        }
      } else {
        if (!isMainWebSocketConnected) {
          return;
        }
        setConversationMode("code");
        const task = trimmed.slice(CODE_COMMAND.length).trim();
        if (task) {
          // The parent (code) conversation always already exists, so this
          // can always send immediately — same synchronous-mode-read routing
          // as the /plan-with-existing-planner case above.
          onSubmit(task);
        }
      }
    },
    [
      conversationId,
      curAgentState,
      hasPlanner,
      isCreatingConversation,
      isMainWebSocketConnected,
      isPlanningAgentRunning,
      isWebSocketConnected,
      onSubmit,
      handlePlanClick,
      setConversationMode,
    ],
  );
};
