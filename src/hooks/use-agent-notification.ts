import { useEffect, useRef } from "react";
import { AgentState } from "#/types/agent-state";
import { useSettings } from "#/hooks/query/use-settings";
import notificationSound from "#/assets/notification.mp3";

const NOTIFICATION_STATES: AgentState[] = [
  AgentState.AWAITING_USER_INPUT,
  AgentState.FINISHED,
  AgentState.AWAITING_USER_CONFIRMATION,
];

/**
 * Hook that plays a notification sound when the agent stops running and now
 * needs the user's attention. Opening, reloading or switching to a
 * conversation that is already idle or finished is not such a transition.
 * The browser tab title itself is managed by `useAppTitle`, which prefixes
 * the title with an emoji that reflects the current agent state.
 */
export function useAgentNotification(
  curAgentState: AgentState,
  conversationId?: string | null,
) {
  const { data: settings } = useSettings();
  const audioRef = useRef<HTMLAudioElement | undefined>(undefined);
  const prevRef = useRef<{
    conversationId?: string | null;
    state: AgentState;
  }>(undefined);

  // Initialize audio only in browser environment, inside useEffect to
  // avoid side effects during render (React 18 strict mode, SSR safety).
  useEffect(() => {
    if (typeof window !== "undefined" && !audioRef.current) {
      audioRef.current = new Audio(notificationSound);
      audioRef.current.volume = 0.5;
    }
  }, []);

  const isSoundEnabled = settings?.enable_sound_notifications ?? false;

  // Trigger notification only when this conversation's agent leaves RUNNING
  // for a notification-worthy state — not when unrelated deps (e.g. settings)
  // change.
  useEffect(() => {
    const prev = prevRef.current;
    prevRef.current = { conversationId, state: curAgentState };

    if (
      prev?.state !== AgentState.RUNNING ||
      prev.conversationId !== conversationId ||
      !NOTIFICATION_STATES.includes(curAgentState)
    )
      return;

    if (isSoundEnabled && audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play().catch(() => {
        // Ignore autoplay errors (browsers may block autoplay)
      });
    }
  }, [curAgentState, conversationId, isSoundEnabled]);
}
