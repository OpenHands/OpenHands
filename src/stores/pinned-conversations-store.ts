import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

interface PinnedConversationsState {
  pinsByBackendId: Record<string, string[]>;
}

interface PinnedConversationsActions {
  pinConversation: (backendId: string, conversationId: string) => void;
  unpinConversation: (backendId: string, conversationId: string) => void;
  togglePin: (backendId: string, conversationId: string) => void;
  pruneMissingConversations: (
    backendId: string,
    existingIds: readonly string[],
  ) => void;
}

type PinnedConversationsStore = PinnedConversationsState &
  PinnedConversationsActions;

const initialState: PinnedConversationsState = {
  pinsByBackendId: {},
};

/**
 * Scope key a backend's pins are stored under.
 *
 * Pins follow the conversation list they were set from: local backends have
 * no org, so their pins keep the bare backend id (and stay readable for
 * anyone who pinned before pins became org-aware). Cloud backends are
 * org-scoped — a conversation pinned in one org must not show as pinned
 * while a different org is active.
 */
export function getPinnedConversationsScopeKey(
  backendId: string,
  orgId?: string | null,
): string {
  return orgId ? `${backendId}:${orgId}` : backendId;
}

function getPinsForBackend(
  pinsByBackendId: Record<string, string[]>,
  backendId: string,
): string[] {
  return pinsByBackendId[backendId] ?? [];
}

export const usePinnedConversationsStore = create<PinnedConversationsStore>()(
  persist(
    (set, get) => ({
      ...initialState,

      pinConversation: (backendId, conversationId) => {
        const current = getPinsForBackend(get().pinsByBackendId, backendId);
        if (current.includes(conversationId)) {
          return;
        }
        set((state) => ({
          pinsByBackendId: {
            ...state.pinsByBackendId,
            [backendId]: [conversationId, ...current],
          },
        }));
      },

      unpinConversation: (backendId, conversationId) => {
        const current = getPinsForBackend(get().pinsByBackendId, backendId);
        if (!current.includes(conversationId)) {
          return;
        }
        set((state) => ({
          pinsByBackendId: {
            ...state.pinsByBackendId,
            [backendId]: current.filter((id) => id !== conversationId),
          },
        }));
      },

      togglePin: (backendId, conversationId) => {
        const current = getPinsForBackend(get().pinsByBackendId, backendId);
        if (current.includes(conversationId)) {
          get().unpinConversation(backendId, conversationId);
        } else {
          get().pinConversation(backendId, conversationId);
        }
      },

      pruneMissingConversations: (backendId, existingIds) => {
        const existing = new Set(existingIds);
        const current = getPinsForBackend(get().pinsByBackendId, backendId);
        const pruned = current.filter((id) => existing.has(id));
        if (pruned.length === current.length) {
          return;
        }
        set((state) => ({
          pinsByBackendId: {
            ...state.pinsByBackendId,
            [backendId]: pruned,
          },
        }));
      },
    }),
    {
      name: "pinned-conversations",
      storage: createJSONStorage(() => localStorage),
      partialize: (state): PinnedConversationsState => ({
        pinsByBackendId: state.pinsByBackendId,
      }),
    },
  ),
);
