import { create } from "zustand";
import {
  isAgentNotificationsStagingEnabled,
  isStagedAgentNotificationId,
  type AgentNotification,
} from "#/components/features/chat/agent-notifications.constants";
import {
  readAgentNotificationsHistory,
  readSeenAgentNotificationIds,
  writeAgentNotificationsHistory,
  writeSeenAgentNotificationIds,
} from "#/components/features/chat/agent-notifications-storage";

function readSanitizedHistory(conversationId: string): AgentNotification[] {
  const history = readAgentNotificationsHistory(conversationId);
  if (isAgentNotificationsStagingEnabled()) {
    return history;
  }

  const withoutStaged = history.filter(
    (notification) => !isStagedAgentNotificationId(notification.id),
  );
  if (withoutStaged.length !== history.length) {
    writeAgentNotificationsHistory(conversationId, withoutStaged);
  }
  return withoutStaged;
}

function readSanitizedSeenIds(conversationId: string): string[] {
  const seenIds = readSeenAgentNotificationIds(conversationId);
  if (isAgentNotificationsStagingEnabled()) {
    return seenIds;
  }

  const withoutStaged = seenIds.filter(
    (id) => !isStagedAgentNotificationId(id),
  );
  if (withoutStaged.length !== seenIds.length) {
    writeSeenAgentNotificationIds(conversationId, withoutStaged);
  }
  return withoutStaged;
}

interface AgentNotificationsStoreState {
  historyByConversation: Record<string, AgentNotification[]>;
  seenByConversation: Record<string, string[]>;
  /** Loads a conversation's persisted state into memory the first time it's touched. */
  ensureHydrated: (conversationId: string) => void;
  /** Appends newly-detected notifications, deduped by id, and persists them. */
  addNotifications: (
    conversationId: string,
    incoming: AgentNotification[],
  ) => void;
  /** Marks ids as having been surfaced above the chat input at least once. */
  markSeen: (conversationId: string, ids: string[]) => void;
  /** Permanently removes one notification from history and seen state. */
  removeNotification: (conversationId: string, id: string) => void;
}

export const useAgentNotificationsStore = create<AgentNotificationsStoreState>(
  (set, get) => ({
    historyByConversation: {},
    seenByConversation: {},

    ensureHydrated: (conversationId) => {
      if (!conversationId || conversationId in get().historyByConversation) {
        return;
      }

      set((state) => ({
        historyByConversation: {
          ...state.historyByConversation,
          [conversationId]: readSanitizedHistory(conversationId),
        },
        seenByConversation: {
          ...state.seenByConversation,
          [conversationId]: readSanitizedSeenIds(conversationId),
        },
      }));
    },

    addNotifications: (conversationId, incoming) => {
      if (!conversationId || incoming.length === 0) {
        return;
      }

      get().ensureHydrated(conversationId);

      set((state) => {
        const existing = state.historyByConversation[conversationId] ?? [];
        const existingIds = new Set(existing.map((n) => n.id));
        const additions = incoming.filter((n) => !existingIds.has(n.id));
        if (additions.length === 0) {
          return state;
        }

        const updated = [...existing, ...additions];
        writeAgentNotificationsHistory(conversationId, updated);

        return {
          historyByConversation: {
            ...state.historyByConversation,
            [conversationId]: updated,
          },
        };
      });
    },

    markSeen: (conversationId, ids) => {
      if (!conversationId || ids.length === 0) {
        return;
      }

      get().ensureHydrated(conversationId);

      set((state) => {
        const seen = new Set(state.seenByConversation[conversationId] ?? []);
        ids.forEach((id) => seen.add(id));
        const updated = Array.from(seen);
        writeSeenAgentNotificationIds(conversationId, updated);

        return {
          seenByConversation: {
            ...state.seenByConversation,
            [conversationId]: updated,
          },
        };
      });
    },

    removeNotification: (conversationId, id) => {
      if (!conversationId || !id) {
        return;
      }

      get().ensureHydrated(conversationId);

      set((state) => {
        const existing = state.historyByConversation[conversationId] ?? [];
        const updatedHistory = existing.filter(
          (notification) => notification.id !== id,
        );
        if (updatedHistory.length === existing.length) {
          return state;
        }

        writeAgentNotificationsHistory(conversationId, updatedHistory);

        const updatedSeen = (
          state.seenByConversation[conversationId] ?? []
        ).filter((seenId) => seenId !== id);
        writeSeenAgentNotificationIds(conversationId, updatedSeen);

        return {
          historyByConversation: {
            ...state.historyByConversation,
            [conversationId]: updatedHistory,
          },
          seenByConversation: {
            ...state.seenByConversation,
            [conversationId]: updatedSeen,
          },
        };
      });
    },
  }),
);
