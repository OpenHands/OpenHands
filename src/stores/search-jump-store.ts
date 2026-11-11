import { create } from "zustand";

/**
 * One-shot "jump to this message" state, set when the user clicks a content
 * search result and consumed by the chat once the target event is rendered.
 * Events can arrive long after navigation (streaming, lazy history load),
 * so the target stays pending until found or the user navigates away.
 */
interface SearchJumpState {
  /** Event id to scroll to; null when no jump is pending. */
  targetEventId: string | null;
  /** Words to highlight inside the target message (the search query). */
  highlightTerms: string[];
  /** Fire a jump; the chat clears it once handled. */
  startJump: (eventId: string, highlightTerms: string[]) => void;
  /** Clear the pending jump (handled, cancelled, or stale). */
  clearJump: () => void;
}

const MAX_HIGHLIGHT_TERMS = 8;

export const useSearchJumpStore = create<SearchJumpState>((set) => ({
  targetEventId: null,
  highlightTerms: [],
  startJump: (eventId, highlightTerms) =>
    set({
      targetEventId: eventId,
      highlightTerms: highlightTerms.slice(0, MAX_HIGHLIGHT_TERMS),
    }),
  clearJump: () => set({ targetEventId: null, highlightTerms: [] }),
}));
