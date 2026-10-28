import { create } from "zustand";
import type { ReplyQuote } from "#/utils/message-quote";
import {
  getConversationState,
  setConversationState,
} from "#/utils/conversation-local-storage";

/**
 * The reply quote armed in the composer, tagged with the conversation it
 * belongs to. Mirrors the `conversationId` tagging of
 * `PendingUserMessage` so switching conversations never carries an armed
 * quote over.
 */
export interface ArmedReplyQuote extends ReplyQuote {
  conversationId: string;
}

interface ReplyQuoteState {
  pendingReplyQuote: ArmedReplyQuote | null;
}

interface ReplyQuoteActions {
  /**
   * Arm the composer's quote card for a conversation. Sole writer of the
   * Reply action; persists the quote next to the conversation draft so a
   * mid-compose reload survives. Persistence is skipped for task /
   * temporary conversation ids by `setConversationState` itself.
   */
  setReplyQuote: (conversationId: string, quote: ReplyQuote) => void;
  /**
   * Disarm the quote card (dismiss or submit). Sole writer of clearing; the
   * persisted copy is removed with the same call.
   */
  clearReplyQuote: (conversationId: string) => void;
  /**
   * Re-arm the quote persisted in conversation localStorage for
   * `conversationId`. Called on conversation mount: a different
   * conversation (or one without a persisted quote) clears the armed
   * state; the same conversation with a persisted quote restores it.
   */
  hydrateReplyQuote: (conversationId: string) => void;
}

type ReplyQuoteStore = ReplyQuoteState & ReplyQuoteActions;

export const useReplyQuoteStore = create<ReplyQuoteStore>((set, get) => ({
  pendingReplyQuote: null,

  setReplyQuote: (conversationId, quote) => {
    set({ pendingReplyQuote: { ...quote, conversationId } });
    setConversationState(conversationId, { pendingReplyQuote: quote });
  },

  clearReplyQuote: (conversationId) => {
    if (get().pendingReplyQuote === null) return;
    set({ pendingReplyQuote: null });
    setConversationState(conversationId, { pendingReplyQuote: null });
  },

  hydrateReplyQuote: (conversationId) => {
    const armed = get().pendingReplyQuote;
    if (armed?.conversationId === conversationId) return;

    const stored =
      conversationId === ""
        ? null
        : getConversationState(conversationId).pendingReplyQuote;
    set({ pendingReplyQuote: stored ? { ...stored, conversationId } : null });
  },
}));

/**
 * Selector scoped to the active conversation, mirroring how
 * `PendingUserMessages` filters the global queue by conversation id.
 */
export const usePendingReplyQuote = (
  conversationId: string,
): ArmedReplyQuote | null => {
  const armed = useReplyQuoteStore((state) => state.pendingReplyQuote);
  return armed?.conversationId === conversationId ? armed : null;
};
