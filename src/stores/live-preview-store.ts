import { create } from "zustand";

interface LivePreviewState {
  requestedPaths: Record<string, string | undefined>;
  setRequestedPath: (conversationId: string, path: string) => void;
  clearRequestedPath: (conversationId: string) => void;
}

/**
 * Holds the last preview path requested by the agent for each conversation.
 * This is intentionally ephemeral: the workspace is the source of truth and
 * a stale path should never survive into an unrelated conversation.
 */
export const useLivePreviewStore = create<LivePreviewState>((set) => ({
  requestedPaths: {},
  setRequestedPath: (conversationId, path) =>
    set((state) => ({
      requestedPaths: { ...state.requestedPaths, [conversationId]: path },
    })),
  clearRequestedPath: (conversationId) =>
    set((state) => {
      const requestedPaths = { ...state.requestedPaths };
      delete requestedPaths[conversationId];
      return { requestedPaths };
    }),
}));
