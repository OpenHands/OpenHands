import { create } from "zustand";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

/**
 * Live runtime fields for one conversation. Buckets are keyed by conversation
 * id so a popout and the primary route can each keep their own execution
 * status without leaking typing/confirmation state across windows.
 */
export interface ConversationRuntimeBucket {
  execution_status: ExecutionStatus | null;
}

interface ConversationStateStore {
  byConversation: Record<string, ConversationRuntimeBucket>;

  /**
   * Set the agent execution status for a conversation.
   */
  setExecutionStatus: (
    conversationId: string,
    execution_status: ExecutionStatus,
  ) => void;

  /**
   * Drop one conversation's runtime bucket (e.g. popout closed, route reset).
   */
  clearConversation: (conversationId: string) => void;

  /**
   * Reset every conversation's runtime state (tests / full teardown).
   */
  reset: () => void;
}

const EMPTY_BUCKET: ConversationRuntimeBucket = Object.freeze({
  execution_status: null,
});

export const getConversationExecutionStatus = (
  state: Pick<ConversationStateStore, "byConversation">,
  conversationId: string | null | undefined,
): ExecutionStatus | null =>
  conversationId
    ? (state.byConversation[conversationId]?.execution_status ?? null)
    : null;

export const useConversationStateStore = create<ConversationStateStore>(
  (set) => ({
    byConversation: {},

    setExecutionStatus: (conversationId, execution_status) =>
      set((state) => ({
        byConversation: {
          ...state.byConversation,
          [conversationId]: { execution_status },
        },
      })),

    clearConversation: (conversationId) =>
      set((state) => {
        if (!(conversationId in state.byConversation)) {
          return state;
        }
        const { [conversationId]: _removed, ...byConversation } =
          state.byConversation;
        return { byConversation };
      }),

    reset: () => set({ byConversation: {} }),
  }),
);

/** Stable empty for selectors that need a frozen fallback reference. */
export const EMPTY_RUNTIME_BUCKET = EMPTY_BUCKET;
