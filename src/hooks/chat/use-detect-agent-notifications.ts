import { useCallback } from "react";
import { collectAgentNotificationsFromEvents } from "#/components/features/chat/collect-agent-notifications-from-events";
import { useEventStore } from "#/stores/use-event-store";
import { useAgentNotificationsStore } from "#/stores/use-agent-notifications-store";

export interface DetectAgentNotificationsResult {
  found: number;
  added: number;
}

/**
 * Manual and programmatic entry point for scanning the loaded conversation
 * events and merging recommendations into the notifications store.
 */
export function useDetectAgentNotifications(conversationId: string | null) {
  const ensureHydrated = useAgentNotificationsStore(
    (state) => state.ensureHydrated,
  );
  const addNotifications = useAgentNotificationsStore(
    (state) => state.addNotifications,
  );

  const detectNow = useCallback((): DetectAgentNotificationsResult => {
    if (!conversationId) {
      return { found: 0, added: 0 };
    }

    ensureHydrated(conversationId);

    const detected = collectAgentNotificationsFromEvents(
      useEventStore.getState().events,
    );
    const historyBefore =
      useAgentNotificationsStore.getState().historyByConversation[
        conversationId
      ]?.length ?? 0;

    if (detected.length > 0) {
      addNotifications(conversationId, detected);
    }

    const historyAfter =
      useAgentNotificationsStore.getState().historyByConversation[
        conversationId
      ]?.length ?? 0;

    return {
      found: detected.length,
      added: historyAfter - historyBefore,
    };
  }, [addNotifications, conversationId, ensureHydrated]);

  return { detectNow };
}
