import { useCallback } from "react";
import { useConversationStore } from "#/stores/conversation-store";
import { useHandlePlanClick } from "#/hooks/use-handle-plan-click";
import { useUnifiedWebSocketStatus } from "#/hooks/use-unified-websocket-status";
import { AgentState } from "#/types/agent-state";
import { PLAN_COMMAND, CODE_COMMAND } from "#/utils/constants";

const PLAN_PREFIX = `${PLAN_COMMAND} `;

/**
 * Intercepts "/plan [task]" and "/code" submissions and toggles the
 * conversation's mode the same way the Code/Plan button does, instead of
 * sending them as a chat message. Lets mode be switched (or turned off)
 * without needing that button to be reachable. Everything else falls through
 * to `onSubmit`. Passthrough when `conversationId` is null.
 *
 * A bare "/plan" only switches mode, matching the toggle button. "/plan
 * <task>" additionally delivers `<task>` to the planner: immediately via
 * `onSubmit` if a planner helper already exists (mode is set synchronously
 * before `onSubmit` runs, so the send routes to it — see
 * conversation-websocket-context.tsx's `getState().conversationMode` read),
 * otherwise as the new planner's `initial_message` so it isn't lost while
 * creation is in flight.
 *
 * Swallows the command (no toggle, no chat message) while the agent is
 * running, a planning conversation is already being created, or the
 * websocket is disconnected — the same conditions that disable the
 * Code/Plan button — since flipping the mode mid-run, or before the socket
 * can carry it, would leave it out of sync with which conversation the
 * in-flight run is actually targeting.
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
  const isWebSocketConnected = useUnifiedWebSocketStatus() === "OPEN";

  return useCallback(
    (message: string) => {
      const trimmed = message.trim();
      const isPlan =
        trimmed === PLAN_COMMAND || trimmed.startsWith(PLAN_PREFIX);
      const isCode = trimmed === CODE_COMMAND;
      if (!conversationId || (!isPlan && !isCode)) {
        onSubmit(message);
        return;
      }

      if (
        curAgentState === AgentState.RUNNING ||
        isCreatingConversation ||
        !isWebSocketConnected
      ) {
        return;
      }

      if (isPlan) {
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
        setConversationMode("code");
      }
    },
    [
      conversationId,
      curAgentState,
      hasPlanner,
      isCreatingConversation,
      isWebSocketConnected,
      onSubmit,
      handlePlanClick,
      setConversationMode,
    ],
  );
};
