import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

export interface ConversationReadState {
  /** Map of conversationId -> ISO string timestamp when conversation was last read */
  lastReadAt: Record<string, string>;
  /** Map of conversationId -> boolean when user explicitly marked as unread */
  manualUnread: Record<string, boolean>;
}

export interface IsConversationUnreadOptions {
  conversationId: string;
  lastUpdatedAt?: string | null;
  executionStatus?: ExecutionStatus | null;
  isCurrent?: boolean;
}

export interface ConversationReadStateActions {
  markAsRead: (conversationId: string) => void;
  markAsUnread: (conversationId: string) => void;
  markAllAsRead: (conversationIds: string[]) => void;
  isConversationUnread: (options: IsConversationUnreadOptions) => boolean;
  /**
   * Prunes entries for conversations that no longer exist in the backend.
   * Mirrors the pattern in pinned-conversations-store.ts to keep localStorage bounded.
   */
  pruneMissingConversations: (existingIds: readonly string[]) => void;
}

export type ConversationReadStateStore = ConversationReadState &
  ConversationReadStateActions;

export const INITIAL_CONVERSATION_READ_STATE: ConversationReadState = {
  lastReadAt: {},
  manualUnread: {},
};

export const useConversationReadStateStore =
  create<ConversationReadStateStore>()(
    persist(
      (set, get) => ({
        ...INITIAL_CONVERSATION_READ_STATE,

        markAsRead: (conversationId: string) => {
          const now = new Date().toISOString();
          set((state) => {
            const nextManualUnread = { ...state.manualUnread };
            delete nextManualUnread[conversationId];
            return {
              lastReadAt: {
                ...state.lastReadAt,
                [conversationId]: now,
              },
              manualUnread: nextManualUnread,
            };
          });
        },

        markAsUnread: (conversationId: string) => {
          set((state) => ({
            manualUnread: {
              ...state.manualUnread,
              [conversationId]: true,
            },
          }));
        },

        markAllAsRead: (conversationIds: string[]) => {
          const now = new Date().toISOString();
          set((state) => {
            const nextLastReadAt = { ...state.lastReadAt };
            const nextManualUnread = { ...state.manualUnread };
            for (const id of conversationIds) {
              nextLastReadAt[id] = now;
              delete nextManualUnread[id];
            }
            return {
              lastReadAt: nextLastReadAt,
              manualUnread: nextManualUnread,
            };
          });
        },

        pruneMissingConversations: (existingIds: readonly string[]) => {
          const existing = new Set(existingIds);
          set((state) => {
            let hasChanges = false;
            const nextLastReadAt: Record<string, string> = {};
            for (const [id, val] of Object.entries(state.lastReadAt)) {
              if (existing.has(id)) {
                nextLastReadAt[id] = val;
              } else {
                hasChanges = true;
              }
            }
            const nextManualUnread: Record<string, boolean> = {};
            for (const [id, val] of Object.entries(state.manualUnread)) {
              if (existing.has(id)) {
                nextManualUnread[id] = val;
              } else {
                hasChanges = true;
              }
            }
            if (!hasChanges) {
              return state;
            }
            return {
              lastReadAt: nextLastReadAt,
              manualUnread: nextManualUnread,
            };
          });
        },

        isConversationUnread: ({
          conversationId,
          lastUpdatedAt,
          executionStatus,
          isCurrent,
        }: IsConversationUnreadOptions): boolean => {
          if (isCurrent) {
            return false;
          }

          const { manualUnread, lastReadAt } = get();

          // Explicit manual override
          if (manualUnread[conversationId] === true) {
            return true;
          }

          // Smart stream tracking: if execution is currently RUNNING, it's actively working,
          // not ready to be flagged as unread until the run settles
          if (executionStatus === ExecutionStatus.RUNNING) {
            return false;
          }

          if (!lastUpdatedAt) {
            return false;
          }

          const readAt = lastReadAt[conversationId];
          if (!readAt) {
            // Never opened/read
            return true;
          }

          return new Date(lastUpdatedAt).getTime() > new Date(readAt).getTime();
        },
      }),
      {
        name: "openhands-conversation-read-state",
        storage: createJSONStorage(() => localStorage),
        partialize: (state): ConversationReadState => ({
          lastReadAt: state.lastReadAt,
          manualUnread: state.manualUnread,
        }),
      },
    ),
  );
