import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

interface PinnedConversationsState {
  pinsByBackendId: Record<string, string[]>;
}

interface PinnedConversationsActions {
  pinConversation: (
    backendId: string,
    conversationId: string,
    legacyBackendId?: string,
  ) => void;
  unpinConversation: (
    backendId: string,
    conversationId: string,
    legacyBackendId?: string,
  ) => void;
  togglePin: (
    backendId: string,
    conversationId: string,
    legacyBackendId?: string,
  ) => void;
  pruneMissingConversations: (
    backendId: string,
    existingIds: readonly string[],
    legacyBackendId?: string,
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

const EMPTY_PINNED_IDS: string[] = [];

/** Scoped writes override the legacy list, including an explicitly empty list. */
export function getPinnedConversationIds(
  pinsByBackendId: Record<string, string[]>,
  backendId: string,
  legacyBackendId?: string,
): string[] {
  return (
    pinsByBackendId[backendId] ??
    pinsByBackendId[legacyBackendId ?? backendId] ??
    EMPTY_PINNED_IDS
  );
}

export const usePinnedConversationsStore = create<PinnedConversationsStore>()(
  persist(
    (set, get) => ({
      ...initialState,

      pinConversation: (backendId, conversationId, legacyBackendId) => {
        const current = getPinnedConversationIds(
          get().pinsByBackendId,
          backendId,
          legacyBackendId,
        );
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

      unpinConversation: (backendId, conversationId, legacyBackendId) => {
        const current = getPinnedConversationIds(
          get().pinsByBackendId,
          backendId,
          legacyBackendId,
        );
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

      togglePin: (backendId, conversationId, legacyBackendId) => {
        const current = getPinnedConversationIds(
          get().pinsByBackendId,
          backendId,
          legacyBackendId,
        );
        if (current.includes(conversationId)) {
          get().unpinConversation(backendId, conversationId, legacyBackendId);
        } else {
          get().pinConversation(backendId, conversationId, legacyBackendId);
        }
      },

      pruneMissingConversations: (backendId, existingIds, legacyBackendId) => {
        const existing = new Set(existingIds);
        const current = getPinnedConversationIds(
          get().pinsByBackendId,
          backendId,
          legacyBackendId,
        );
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
