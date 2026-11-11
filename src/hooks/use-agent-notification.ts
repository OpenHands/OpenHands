import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { AgentState } from "#/types/agent-state";
import { useSettings } from "#/hooks/query/use-settings";
import { isPlaceholderConversationTitle } from "#/utils/conversation-title";
import { I18nKey } from "#/i18n/declaration";
import notificationSound from "#/assets/notification.mp3";

const NOTIFICATION_STATES: AgentState[] = [
  AgentState.AWAITING_USER_INPUT,
  AgentState.FINISHED,
  AgentState.AWAITING_USER_CONFIRMATION,
];

/** Context used to identify the conversation in a desktop notification. */
export interface AgentNotificationContext {
  /** Coalesces repeat notifications for the same conversation via `tag`. */
  conversationId?: string | null;
  /** Shown as the notification title; falls back to an untitled label. */
  conversationTitle?: string | null;
}

/**
 * The body shown for each attention-worthy state. Kept beside the state list
 * so the two stay in sync.
 */
function getDesktopNotificationBody(state: AgentState): I18nKey {
  switch (state) {
    case AgentState.AWAITING_USER_INPUT:
      return I18nKey.NOTIFICATION$AWAITING_INPUT;
    case AgentState.FINISHED:
      return I18nKey.NOTIFICATION$FINISHED;
    default:
      return I18nKey.NOTIFICATION$CONFIRMATION;
  }
}

/**
 * Hook that reacts to the agent transitioning into a state that requires
 * user attention. Two additive channels, both opt-in:
 *
 * - a notification sound (existing `enable_sound_notifications` setting);
 * - a desktop (Web Notification API) notification (opt-in
 *   `enable_desktop_notifications`), shown only while the tab is backgrounded
 *   so the in-tab UI is not shadowed. Clicking it refocuses the conversation
 *   and dismisses it.
 *
 * The browser tab title itself is managed by `useAppTitle`, which prefixes
 * the title with an emoji that reflects the current agent state.
 */
export function useAgentNotification(
  curAgentState: AgentState,
  { conversationId, conversationTitle }: AgentNotificationContext = {},
) {
  const { t } = useTranslation("openhands");
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
  const isDesktopEnabled = settings?.enable_desktop_notifications ?? false;

  // Trigger notifications only on actual state transitions into a
  // notification-worthy state — not when unrelated deps (e.g. settings)
  // change. Re-reading permission at fire time keeps a permission grant made
  // after mount effective without storing browser state in React.
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

    if (
      isDesktopEnabled &&
      typeof Notification !== "undefined" &&
      Notification.permission === "granted" &&
      document.hidden
    ) {
      const hasRealTitle =
        !!conversationTitle &&
        !isPlaceholderConversationTitle(
          conversationTitle,
          conversationId ?? "",
        );
      const title = hasRealTitle
        ? conversationTitle
        : t(I18nKey.NOTIFICATION$UNTITLED_CONVERSATION);
      const notification = new Notification(title, {
        body: t(getDesktopNotificationBody(curAgentState)),
        tag: conversationId ?? undefined,
      });
      notification.onclick = () => {
        window.focus();
        notification.close();
      };
    }
  }, [
    curAgentState,
    isSoundEnabled,
    isDesktopEnabled,
    conversationId,
    conversationTitle,
    t,
  ]);
}
