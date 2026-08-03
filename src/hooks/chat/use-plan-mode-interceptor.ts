import { useCallback } from "react";
import { useConversationStore } from "#/stores/conversation-store";
import { useHandlePlanClick } from "#/hooks/use-handle-plan-click";
import { useUnifiedWebSocketStatus } from "#/hooks/use-unified-websocket-status";
import { AgentState } from "#/types/agent-state";
import { PLAN_COMMAND, CODE_COMMAND } from "#/utils/constants";

/**
 * Intercepts "/plan" and "/code" submissions and toggles the conversation's
 * mode the same way the Code/Plan button does, instead of sending them as a
 * chat message. Lets mode be switched (or turned off) without needing that
 * button to be reachable. Everything else falls through to `onSubmit`.
 * Passthrough when `conversationId` is null.
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
  const { handlePlanClick, isCreatingConversation } = useHandlePlanClick();
  const isWebSocketConnected = useUnifiedWebSocketStatus() === "OPEN";

  return useCallback(
    (message: string) => {
      const trimmed = message.trim();
      const isPlan = trimmed === PLAN_COMMAND;
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
        handlePlanClick();
      } else {
        setConversationMode("code");
      }
    },
    [
      conversationId,
      curAgentState,
      isCreatingConversation,
      isWebSocketConnected,
      onSubmit,
      handlePlanClick,
      setConversationMode,
    ],
  );
};
