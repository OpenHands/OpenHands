import { useEffect } from "react";
import { useConversationStore } from "#/stores/conversation-store";

/** Grace period so the pointer can move from the toggle onto the peek panel. */
export const CONVERSATION_OVERVIEW_PEEK_CLOSE_DELAY_MS = 150;

let peekCloseTimer: ReturnType<typeof setTimeout> | null = null;

function clearPeekCloseTimer() {
  if (peekCloseTimer !== null) {
    clearTimeout(peekCloseTimer);
    peekCloseTimer = null;
  }
}

export function openConversationOverviewPanelPeek() {
  const { isRightPanelShown, isOverviewPanelShown, setIsOverviewPanelPeeked } =
    useConversationStore.getState();

  if (!isRightPanelShown || isOverviewPanelShown) {
    return;
  }

  clearPeekCloseTimer();
  setIsOverviewPanelPeeked(true);
}

export function scheduleCloseConversationOverviewPanelPeek() {
  clearPeekCloseTimer();
  peekCloseTimer = setTimeout(() => {
    peekCloseTimer = null;
    useConversationStore.getState().setIsOverviewPanelPeeked(false);
  }, CONVERSATION_OVERVIEW_PEEK_CLOSE_DELAY_MS);
}

export function closeConversationOverviewPanelPeek() {
  clearPeekCloseTimer();
  useConversationStore.getState().setIsOverviewPanelPeeked(false);
}

/**
 * Clears a stale peek when the files drawer closes or overview is pinned open.
 */
export function useSyncConversationOverviewPanelPeek() {
  const isRightPanelShown = useConversationStore(
    (state) => state.isRightPanelShown,
  );
  const isOverviewPanelShown = useConversationStore(
    (state) => state.isOverviewPanelShown,
  );
  const isOverviewPanelPeeked = useConversationStore(
    (state) => state.isOverviewPanelPeeked,
  );

  useEffect(() => {
    if (isOverviewPanelPeeked && (!isRightPanelShown || isOverviewPanelShown)) {
      closeConversationOverviewPanelPeek();
    }
  }, [isOverviewPanelPeeked, isOverviewPanelShown, isRightPanelShown]);
}
