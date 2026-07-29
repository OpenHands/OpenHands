import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { AgentState } from "#/types/agent-state";
import { useSettings } from "#/hooks/query/use-settings";
import notificationSound from "#/assets/notification.mp3";
import { I18nKey } from "#/i18n/declaration";
import { useNavigation } from "#/context/navigation-context";

const NOTIFICATION_STATES: AgentState[] = [
  AgentState.AWAITING_USER_INPUT,
  AgentState.FINISHED,
  AgentState.AWAITING_USER_CONFIRMATION,
];

const getNotificationBodyKey = (state: AgentState): I18nKey | null => {
  switch (state) {
    case AgentState.FINISHED:
      return I18nKey.DESKTOP_NOTIFICATIONS$FINISHED;
    case AgentState.AWAITING_USER_INPUT:
      return I18nKey.DESKTOP_NOTIFICATIONS$AWAITING_USER_INPUT;
    case AgentState.AWAITING_USER_CONFIRMATION:
      return I18nKey.DESKTOP_NOTIFICATIONS$AWAITING_USER_CONFIRMATION;
    default:
      return null;
  }
};

/**
 * Hook that alerts the user when the agent transitions into a state that
 * requires attention. The browser tab title itself is managed by
 * `useAppTitle`, which prefixes the title with an emoji that reflects the
 * current agent state.
 */
export function useAgentNotification(
  curAgentState: AgentState,
  conversationTitle?: string | null,
) {
  const { t } = useTranslation("openhands");
  const { conversationId, navigate } = useNavigation();
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
  const areDesktopNotificationsEnabled =
    settings?.enable_desktop_notifications ?? false;

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

    const notificationBodyKey = getNotificationBodyKey(curAgentState);
    if (
      !areDesktopNotificationsEnabled ||
      !notificationBodyKey ||
      !conversationId ||
      typeof Notification === "undefined" ||
      Notification.permission !== "granted" ||
      document.visibilityState === "visible"
    ) {
      return;
    }

    const notification = new Notification(conversationTitle || "OpenHands", {
      body: t(notificationBodyKey),
    });
    notification.onclick = () => {
      window.focus();
      navigate(`/conversations/${conversationId}`);
      notification.close();
    };
  }, [
    areDesktopNotificationsEnabled,
    conversationId,
    conversationTitle,
    curAgentState,
    isSoundEnabled,
    navigate,
    t,
  ]);
}
