import { useEffect, useRef } from "react";
import { AgentState } from "#/types/agent-state";
import { useSettings } from "#/hooks/query/use-settings";
import notificationSound from "#/assets/notification.mp3";
import {
  fireAttentionNotification,
  getNotificationPermission,
  isDocumentInBackground,
  setAttentionBadge,
} from "#/lib/os-notifications";

/**
 * States that warrant user attention and should surface a notification.
 *
 * ERROR is included alongside the three pre-existing states — an unplanned
 * failure is exactly the kind of thing a user should be told about when they
 * are away from the tab. REJECTED / RATE_LIMITED route through ERROR upstream,
 * so they do not need a separate entry.
 */
const ATTENTION_STATES: AgentState[] = [
  AgentState.AWAITING_USER_INPUT,
  AgentState.FINISHED,
  AgentState.AWAITING_USER_CONFIRMATION,
  AgentState.ERROR,
];

function outcomeLabelForState(state: AgentState): string {
  switch (state) {
    case AgentState.FINISHED:
      return "Finished";
    case AgentState.ERROR:
      return "Error";
    case AgentState.AWAITING_USER_INPUT:
      return "Needs your input";
    case AgentState.AWAITING_USER_CONFIRMATION:
      return "Needs confirmation";
    default:
      return "Attention needed";
  }
}

export interface UseAgentNotificationOptions {
  /** Optional conversation id for stable notification tagging. */
  conversationId?: string | null;
  /** Optional conversation title for the notification body. */
  conversationTitle?: string | null;
}

/**
 * Hook that fires attention signals when the agent transitions into a state
 * the user should act on. Two independent channels:
 *
 *  1. **Sound** (existing): gated on `enable_sound_notifications`, plays
 *     `notification.mp3` on each attention transition. Works only when the
 *     browser allows audio in the current context.
 *
 *  2. **OS notification** (new): gated on `enable_os_notifications` AND
 *     `Notification.permission === "granted"`, AND only while the document
 *     is not focused/visible (so the user staring at Canvas does not get a
 *     popup). Uses a per-conversation `tag` so repeated transitions replace
 *     rather than stack.
 *
 * Both channels coexist — the sound setting does not influence OS
 * notifications and vice versa.
 */
export function useAgentNotification(
  curAgentState: AgentState,
  options?: UseAgentNotificationOptions,
) {
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
  const isOsEnabled = settings?.enable_os_notifications ?? false;

  // Trigger notification only on actual state transitions into an
  // attention-worthy state — not when unrelated deps (e.g. settings) change.
  useEffect(() => {
    if (prevStateRef.current === curAgentState) return;
    const prevState = prevStateRef.current;
    prevStateRef.current = curAgentState;

    if (!ATTENTION_STATES.includes(curAgentState)) {
      // Leaving an attention state: if no other attention conversation is
      // waiting, clear the badge. For the single-conversation hook we just
      // drop the badge when *this* conversation moves out of attention.
      setAttentionBadge(0);
      return;
    }

    // Don't double-fire on repeated renders of the same attention state.
    if (prevStateRef.current === prevState) return;

    if (isSoundEnabled && audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play().catch(() => {
        // Ignore autoplay errors (browsers may block autoplay)
      });
    }

    // OS notifications only when the page is not being actively used.
    if (isOsEnabled && isDocumentInBackground()) {
      const conversationId = options?.conversationId ?? "unknown";
      fireAttentionNotification({
        conversationId,
        conversationTitle: options?.conversationTitle ?? "",
        outcome: outcomeLabelForState(curAgentState),
      });
      setAttentionBadge(1);
    }
  }, [
    curAgentState,
    isSoundEnabled,
    isOsEnabled,
    options?.conversationId,
    options?.conversationTitle,
  ]);
}

export { getNotificationPermission };
