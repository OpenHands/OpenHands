import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { AgentState } from "#/types/agent-state";
import { useSettings } from "#/hooks/query/use-settings";
import notificationSound from "#/assets/notification.mp3";
import { I18nKey } from "#/i18n/declaration";
import { useNavigation } from "#/context/navigation-context";
import { useOptionalActiveBackendContext } from "#/contexts/active-backend-context";

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
  // AgentStatus renders in trees without an ActiveBackendProvider, so this
  // must not throw. Without it the notification still fires; only the
  // restore-originating-backend step on click is skipped.
  const activeBackend = useOptionalActiveBackendContext();
  // Depend on these primitives, not on the context object. The provider
  // memoises its value on the whole backend snapshot, so depending on the
  // object would re-run the notification effect on every unrelated backend
  // state change.
  const originatingBackendId = activeBackend?.active.backend.id;
  const originatingOrgId = activeBackend?.active.orgId;
  const setActiveBackend = activeBackend?.setActive;
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
      if (setActiveBackend && originatingBackendId) {
        setActiveBackend(originatingBackendId, originatingOrgId);
      }
      navigate(`/conversations/${conversationId}`);
      notification.close();
    };
  }, [
    areDesktopNotificationsEnabled,
    originatingBackendId,
    originatingOrgId,
    setActiveBackend,
    conversationId,
    conversationTitle,
    curAgentState,
    isSoundEnabled,
    navigate,
    t,
  ]);
}
