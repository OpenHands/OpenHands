import { create } from "zustand";
import { useConversationStateStore } from "#/stores/conversation-state-store";
import { useConversationStore } from "#/stores/conversation-store";
import { useEventStore } from "#/stores/use-event-store";

export type PopoutMode = "expanded" | "minimized";

export interface Popout {
  /** The conversation this popout is showing. */
  conversationId: string;
  /** Display title for the popout chrome. */
  title: string;
  /** Composer text to restore once (edit-message branches). */
  prefillMessage: string | null;
  mode: PopoutMode;
  /** Higher = more recently focused; used for overflow collapse order. */
  openedAt: number;
}

interface OpenPopoutInput {
  conversationId: string;
  title: string;
  prefillMessage?: string | null;
}

interface PopoutState {
  popouts: Popout[];
  openPopout: (input: OpenPopoutInput) => void;
  closePopout: (conversationId: string) => void;
  minimizePopout: (conversationId: string) => void;
  expandPopout: (conversationId: string) => void;
  toggleMinimized: (conversationId: string) => void;
  clearPrefill: (conversationId: string) => void;
  /** True when a popout is open for this conversation. */
  isOpen: (conversationId: string) => boolean;
}

/**
 * Floating popouts docked at the bottom-right of the app. Session-only —
 * they do not persist across reloads. Closing one drops its conversation-
 * scoped event / runtime / composer buckets when the conversation is not the
 * currently routed primary.
 */
export const usePopoutStore = create<PopoutState>()((set, get) => ({
  popouts: [],
  openPopout: ({ conversationId, title, prefillMessage = null }) =>
    set((state) => {
      const existing = state.popouts.find(
        (entry) => entry.conversationId === conversationId,
      );
      if (existing) {
        // Re-focus an already-open window and expand it.
        return {
          popouts: state.popouts.map((entry) =>
            entry.conversationId === conversationId
              ? {
                  ...entry,
                  mode: "expanded" as const,
                  openedAt: Date.now(),
                  prefillMessage: prefillMessage ?? entry.prefillMessage,
                }
              : entry,
          ),
        };
      }
      return {
        popouts: [
          ...state.popouts,
          {
            conversationId,
            title,
            prefillMessage,
            mode: "expanded",
            openedAt: Date.now(),
          },
        ],
      };
    }),
  closePopout: (conversationId) => {
    set((state) => ({
      popouts: state.popouts.filter(
        (entry) => entry.conversationId !== conversationId,
      ),
    }));
    // Free conversation-scoped buckets unless the primary route still owns
    // this id — the primary WS provider will re-seed if the user later
    // navigates there.
    const primaryPath = globalThis.location.pathname;
    const isPrimary =
      primaryPath === `/conversations/${conversationId}` ||
      primaryPath.startsWith(`/conversations/${conversationId}/`);
    if (!isPrimary) {
      useEventStore.getState().clearConversation(conversationId);
      useConversationStateStore.getState().clearConversation(conversationId);
      useConversationStore.getState().clearComposer(conversationId);
    }
  },
  minimizePopout: (conversationId) =>
    set((state) => ({
      popouts: state.popouts.map((entry) =>
        entry.conversationId === conversationId
          ? { ...entry, mode: "minimized" as const }
          : entry,
      ),
    })),
  expandPopout: (conversationId) =>
    set((state) => ({
      popouts: state.popouts.map((entry) =>
        entry.conversationId === conversationId
          ? {
              ...entry,
              mode: "expanded" as const,
              openedAt: Date.now(),
            }
          : entry,
      ),
    })),
  toggleMinimized: (conversationId) => {
    const entry = get().popouts.find(
      (candidate) => candidate.conversationId === conversationId,
    );
    if (!entry) return;
    if (entry.mode === "minimized") {
      get().expandPopout(conversationId);
    } else {
      get().minimizePopout(conversationId);
    }
  },
  clearPrefill: (conversationId) =>
    set((state) => ({
      popouts: state.popouts.map((entry) =>
        entry.conversationId === conversationId
          ? { ...entry, prefillMessage: null }
          : entry,
      ),
    })),
  isOpen: (conversationId) =>
    get().popouts.some((entry) => entry.conversationId === conversationId),
}));

/** Expanded window footprint used for overflow math. */
export const POPOUT_EXPANDED_WIDTH_PX = 380;
/** Minimized pill footprint used for overflow math. */
export const POPOUT_MINIMIZED_WIDTH_PX = 220;
/** Circular trigger that opens the hidden-popout selector. */
export const POPOUT_OVERFLOW_BUTTON_SIZE_PX = 36;
/** Gap between docked popouts. */
export const POPOUT_GAP_PX = 12;
/** Right/bottom inset of the dock from the viewport edge. */
export const POPOUT_DOCK_INSET_PX = 16;
/**
 * Stacking layer for popouts: above page content (z-10/z-20), below
 * drawer backdrops (z-40), dropdowns (z-50), and modals (z-60).
 */
export const POPOUT_Z_INDEX = 30;
