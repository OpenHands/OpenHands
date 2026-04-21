import { useCallback, useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { AgentState } from "#/types/agent-state";
import { browserTab } from "#/utils/browser-tab";
import { useSettings } from "#/hooks/query/use-settings";
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

  // Initialize audio only in browser environment
  if (typeof window !== "undefined" && !audioRef.current) {
    audioRef.current = new Audio(notificationSound);
    audioRef.current.volume = 0.5;
  }

  const playSound = useCallback(() => {
    if (!settings?.enable_sound_notifications || !audioRef.current) return;
    audioRef.current.currentTime = 0;
    audioRef.current.play().catch(() => {
      // Ignore autoplay errors (browsers may block autoplay)
    });
  }, [settings?.enable_sound_notifications]);

  // Trigger notification when agent enters a notification-worthy state
  useEffect(() => {
    if (!NOTIFICATION_STATES.includes(curAgentState)) return;

    playSound();

    if (typeof document !== "undefined" && !document.hasFocus()) {
      const message = t(`STATUS$${curAgentState.toUpperCase()}`);
      browserTab.startNotification(message);
    }
  }, [curAgentState, playSound, t]);

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
