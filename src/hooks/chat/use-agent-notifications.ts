import { useCallback, useEffect, useMemo, useState } from "react";
import { useSendMessage } from "#/hooks/use-send-message";
import { createChatMessage } from "#/services/chat-service";
import {
  STAGED_AGENT_NOTIFICATIONS,
  isAgentNotificationsStagingEnabled,
  type AgentNotification,
} from "#/components/features/chat/agent-notifications.constants";
import { hasAgentNotificationsHistoryEntry } from "#/components/features/chat/agent-notifications-storage";
import { useAgentNotificationsStore } from "#/stores/use-agent-notifications-store";

const EMPTY_AGENT_NOTIFICATIONS: AgentNotification[] = [];
const EMPTY_SEEN_AGENT_NOTIFICATION_IDS: string[] = [];

interface UseAgentNotificationsOptions {
  conversationId: string | null;
  enabled: boolean;
}

/**
 * Surfaces the notifications above the chat input: only ones not yet shown
 * for this conversation (see `agentNotifications`), backed by the full
 * per-conversation history the bell dropdown reads from separately.
 */
export function useAgentNotifications({
  conversationId,
  enabled,
}: UseAgentNotificationsOptions) {
  const { send } = useSendMessage();
  const [isCreating, setIsCreating] = useState(false);

  const ensureHydrated = useAgentNotificationsStore(
    (state) => state.ensureHydrated,
  );
  const addNotifications = useAgentNotificationsStore(
    (state) => state.addNotifications,
  );
  const markSeen = useAgentNotificationsStore((state) => state.markSeen);
  const removeNotification = useAgentNotificationsStore(
    (state) => state.removeNotification,
  );
  const history = useAgentNotificationsStore((state) =>
    conversationId
      ? (state.historyByConversation[conversationId] ??
        EMPTY_AGENT_NOTIFICATIONS)
      : EMPTY_AGENT_NOTIFICATIONS,
  );
  const seenIds = useAgentNotificationsStore((state) =>
    conversationId
      ? (state.seenByConversation[conversationId] ??
        EMPTY_SEEN_AGENT_NOTIFICATION_IDS)
      : EMPTY_SEEN_AGENT_NOTIFICATION_IDS,
  );

  useEffect(() => {
    if (conversationId) {
      ensureHydrated(conversationId);
    }
  }, [conversationId, ensureHydrated]);

  // Demo fallback on first open only. Do not re-seed after the user clears
  // history — an empty persisted entry means they removed everything.
  useEffect(() => {
    if (
      conversationId &&
      isAgentNotificationsStagingEnabled() &&
      history.length === 0 &&
      !hasAgentNotificationsHistoryEntry(conversationId)
    ) {
      addNotifications(conversationId, STAGED_AGENT_NOTIFICATIONS);
    }
  }, [conversationId, history.length, addNotifications]);

  const seenIdSet = useMemo(() => new Set(seenIds), [seenIds]);
  const recentAgentNotifications = useMemo(
    () => history.filter((notification) => !seenIdSet.has(notification.id)),
    [history, seenIdSet],
  );

  const isVisible =
    enabled && !!conversationId && recentAgentNotifications.length > 0;

  const dismiss = useCallback(() => {
    if (!conversationId) {
      return;
    }

    markSeen(
      conversationId,
      recentAgentNotifications.map((notification) => notification.id),
    );
  }, [conversationId, markSeen, recentAgentNotifications]);

  const createAll = useCallback(
    async (selectedIds: string[]) => {
      if (!conversationId || selectedIds.length === 0 || isCreating) {
        return;
      }

      const selected = history.filter((notification) =>
        selectedIds.includes(notification.id),
      );
      if (selected.length === 0) {
        return;
      }

      setIsCreating(true);
      try {
        for (const notification of selected) {
          await send(
            createChatMessage(
              notification.prompt,
              [],
              [],
              new Date().toISOString(),
            ),
          );
        }
        markSeen(conversationId, selectedIds);
      } finally {
        setIsCreating(false);
      }
    },
    [conversationId, history, isCreating, markSeen, send],
  );

  const remove = useCallback(
    (id: string) => {
      if (!conversationId) {
        return;
      }
      removeNotification(conversationId, id);
    },
    [conversationId, removeNotification],
  );

  return {
    agentNotifications: recentAgentNotifications,
    history,
    isVisible,
    dismiss,
    createAll,
    remove,
    isCreating,
  };
}
