import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { AgentState } from "#/types/agent-state";
import { browserTab } from "#/utils/browser-tab";
import { useSettings } from "#/hooks/query/use-settings";
import { AGENT_STATUS_MAP } from "#/utils/status";
import notificationSound from "#/assets/notification.mp3";

const NOTIFICATION_STATES: AgentState[] = [
  AgentState.AWAITING_USER_INPUT,
  AgentState.FINISHED,
  AgentState.AWAITING_USER_CONFIRMATION,
];

/**
 * Hook that triggers browser tab flashing and notification sound
 * when the agent transitions into a state that requires user attention.
 *
 * - Flashes the browser tab title when the tab is not focused.
 * - Plays a notification sound if enabled in settings.
 * - Stops flashing when the user focuses the tab.
 */
export function useAgentNotification(curAgentState: AgentState) {
  const { data: settings } = useSettings();
  const { t } = useTranslation();
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

    if (typeof document !== "undefined" && !document.hasFocus()) {
      const i18nKey = AGENT_STATUS_MAP[curAgentState];
      const message = i18nKey ? t(i18nKey) : curAgentState;
      browserTab.startNotification(message);
    }
  }, [curAgentState, isSoundEnabled, t]);

  // Stop tab notification when window gains focus
  useEffect(() => {
    if (typeof window === "undefined") return undefined;

    const handleFocus = () => {
      browserTab.stopNotification();
    };

    window.addEventListener("focus", handleFocus);
    return () => {
      window.removeEventListener("focus", handleFocus);
      browserTab.stopNotification();
    };
  }, []);
}
