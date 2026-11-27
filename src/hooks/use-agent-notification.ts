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
 * Hook that plays a notification sound when the agent transitions into a
 * state that requires user attention. The browser tab title itself is
 * managed by `useAppTitle`, which prefixes the title with an emoji that
 * reflects the current agent state.
 */
export function useAgentNotification(curAgentState: AgentState) {
  const { data: settings } = useSettings();
  const audioRef = useRef<HTMLAudioElement | undefined>(undefined);
  const prevStateRef = useRef<AgentState | undefined>(undefined);

  // Initialize audio only in browser environment, inside useEffect to
  // avoid side effects during render (React 18 strict mode, SSR safety).
  useEffect(() => {
    if (typeof window !== "undefined" && !audioRef.current) {
      audioRef.current = new Audio(notificationSound);
      audioRef.current.volume = 0.5;
    }
  }, []);

  const isSoundEnabled = settings?.enable_sound_notifications ?? false;

  // Trigger notification only on actual state transitions into a
  // notification-worthy state — not when unrelated deps (e.g. settings) change.
  useEffect(() => {
    if (prevStateRef.current === curAgentState) return;
    prevStateRef.current = curAgentState;

    if (!NOTIFICATION_STATES.includes(curAgentState)) return;

    if (isSoundEnabled && audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play().catch(() => {
        // Ignore autoplay errors (browsers may block autoplay)
      });
    }
  }, [curAgentState, isSoundEnabled]);
}
