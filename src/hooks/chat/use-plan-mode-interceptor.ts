import { useCallback } from "react";
import { useConversationStore } from "#/stores/conversation-store";
import { useHandlePlanClick } from "#/hooks/use-handle-plan-click";
import {
  useMainWebSocketStatus,
  useUnifiedWebSocketStatus,
} from "#/hooks/use-unified-websocket-status";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";
import { PLAN_COMMAND, CODE_COMMAND } from "#/utils/constants";

const PLAN_PREFIX = `${PLAN_COMMAND} `;
const CODE_PREFIX = `${CODE_COMMAND} `;

/**
 * Intercepts "/plan [task]" and "/code [task]" submissions and toggles the
 * conversation's mode the same way the Code/Plan button does, instead of
 * sending them as a chat message. Lets mode be switched (or turned off)
 * without needing that button to be reachable. Everything else falls through
 * to `onSubmit`. Passthrough when `conversationId` is null.
 *
 * A bare "/plan" or "/code" only switches mode, matching the toggle button.
 * "/plan <task>" additionally delivers `<task>` to the planner: immediately
 * via `onSubmit` if a planner helper already exists (mode is set
 * synchronously before `onSubmit` runs, so the send routes to it — see
 * conversation-websocket-context.tsx's `getState().conversationMode` read),
 * otherwise as the new planner's `initial_message` so it isn't lost while
 * creation is in flight. "/code <task>" is simpler — the parent (code)
 * conversation always already exists — so it just switches mode and sends
 * `<task>` the same synchronous way.
 *
 * Swallows the command (no toggle, no chat message) while the relevant agent
 * is running, a planning conversation is already being created, or the
 * socket the command needs is disconnected, since flipping the mode mid-run,
 * or before the socket can carry it, would leave it out of sync with which
 * conversation the in-flight run is actually targeting. "/code" only ever
 * needs the main socket, so it gates on that alone rather than the
 * main+planning merged status — otherwise a momentary planning-socket
 * reconnect would silently swallow a "/code" the main socket could have
 * sent. "/plan <task>" additionally guards on the planner's own running
 * state (mirroring the `isPlanningAgentRunning` check used elsewhere for the
 * same reason) so a new message isn't routed into a planner still mid-run.
 */
export const usePlanModeInterceptor = (
  conversationId: string | null | undefined,
  curAgentState: AgentState,
  onSubmit: (message: string) => void,
) => {
  const setConversationMode = useConversationStore(
    (s) => s.setConversationMode,
  );
  const localPlanningConversationId = useConversationStore(
    (s) => s.localPlanningConversationId,
  );
  const { handlePlanClick, hasPlanner, isCreatingConversation } =
    useHandlePlanClick();
  const isMainWebSocketConnected = useMainWebSocketStatus() === "OPEN";
  const isWebSocketConnected = useUnifiedWebSocketStatus() === "OPEN";
  const { curAgentState: curPlanningAgentState } = useAgentState(
    localPlanningConversationId ?? undefined,
  );
  const isPlanningAgentRunning =
    !!localPlanningConversationId &&
    (curPlanningAgentState === AgentState.RUNNING ||
      curPlanningAgentState === AgentState.LOADING);

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
