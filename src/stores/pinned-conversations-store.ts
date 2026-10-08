import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

export const PINNED_CONVERSATIONS_STORAGE_KEY = "pinned-conversations";

/**
 * Build the stable scope key for pinned conversations.
 *
 * For local backends (orgId is null/undefined), pins are keyed by backendId.
 * For cloud backends, pins are attributed to the active organization
 * (`backendId::orgId`) so pins do not leak across different workspaces or
 * get erroneously pruned when switching between organizations.
 */
export function getPinnedConversationsScopeKey(
  backendId: string,
  orgId?: string | null,
): string {
  return orgId ? `${backendId}::${orgId}` : backendId;
}

interface PinnedConversationsState {
  pinsByBackendId: Record<string, string[]>;
}

interface PinnedConversationsActions {
  pinConversation: (scopeKey: string, conversationId: string) => void;
  unpinConversation: (scopeKey: string, conversationId: string) => void;
  togglePin: (scopeKey: string, conversationId: string) => void;
  pruneMissingConversations: (
    scopeKey: string,
    existingIds: readonly string[],
  ) => void;
}

type PinnedConversationsStore = PinnedConversationsState &
  PinnedConversationsActions;

const initialState: PinnedConversationsState = {
  pinsByBackendId: {},
};

function getPinsForScope(
  pinsByBackendId: Record<string, string[]>,
  scopeKey: string,
): string[] {
  return pinsByBackendId[scopeKey] ?? [];
}

export const usePinnedConversationsStore = create<PinnedConversationsStore>()(
  persist(
    (set, get) => ({
      ...initialState,

      pinConversation: (scopeKey, conversationId) => {
        const current = getPinsForScope(get().pinsByBackendId, scopeKey);
        if (current.includes(conversationId)) {
          return;
        }
        set((state) => ({
          pinsByBackendId: {
            ...state.pinsByBackendId,
            [scopeKey]: [conversationId, ...current],
          },
        }));
      },

      unpinConversation: (scopeKey, conversationId) => {
        const current = getPinsForScope(get().pinsByBackendId, scopeKey);
        if (!current.includes(conversationId)) {
          return;
        }
        set((state) => ({
          pinsByBackendId: {
            ...state.pinsByBackendId,
            [scopeKey]: current.filter((id) => id !== conversationId),
          },
        }));
      },

      togglePin: (scopeKey, conversationId) => {
        const current = getPinsForScope(get().pinsByBackendId, scopeKey);
        if (current.includes(conversationId)) {
          get().unpinConversation(scopeKey, conversationId);
        } else {
          get().pinConversation(scopeKey, conversationId);
        }
      },

      pruneMissingConversations: (scopeKey, existingIds) => {
        const existing = new Set(existingIds);
        const current = getPinsForScope(get().pinsByBackendId, scopeKey);
        const pruned = current.filter((id) => existing.has(id));
        if (pruned.length === current.length) {
          return;
        }
        set((state) => ({
          pinsByBackendId: {
            ...state.pinsByBackendId,
            [scopeKey]: pruned,
          },
        }));
      },
    }),
    {
      name: PINNED_CONVERSATIONS_STORAGE_KEY,
      storage: createJSONStorage(() => localStorage),
      partialize: (state): PinnedConversationsState => ({
        pinsByBackendId: state.pinsByBackendId,
      }),
    },
  ),
);
