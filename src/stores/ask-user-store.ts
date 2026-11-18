import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import type { AskUserOption } from "#/types/agent-server/core/base/action";

export interface AskUserDecisionRequest {
  conversationId: string;
  toolCallId: string;
  question: string;
  options: AskUserOption[];
  recommended?: string | null;
  /**
   * Settled state: null while pending user decision, string once the user
   * selects an option or submits a response.
   */
  selectedAnswer?: string | null;
}

export interface AskUserState {
  /** Pending or settled requests; each is shown as a decision card in the chat. */
  requests: AskUserDecisionRequest[];
}

export interface AskUserActions {
  enqueue: (request: AskUserDecisionRequest) => void;
  answer: (toolCallId: string, answer: string) => void;
  dismiss: (toolCallId: string) => void;
  clear: () => void;
}

// Persisted so a reload preserves decisions and keeps pending requests open:
// its tool call is already claimed, so replayed ActionEvents will not duplicate it.
export const useAskUserStore = create<AskUserState & AskUserActions>()(
  persist(
    (set) => ({
      requests: [],
      enqueue: (request) =>
        set((state) => {
          if (state.requests.some((r) => r.toolCallId === request.toolCallId)) {
            return state;
          }
          return { requests: [...state.requests, request] };
        }),
      answer: (toolCallId, answer) =>
        set((state) => ({
          requests: state.requests.map((r) =>
            r.toolCallId === toolCallId ? { ...r, selectedAnswer: answer } : r,
          ),
        })),
      dismiss: (toolCallId) =>
        set((state) => ({
          requests: state.requests.filter((r) => r.toolCallId !== toolCallId),
        })),
      clear: () => set({ requests: [] }),
    }),
    {
      name: "openhands-pending-ask-user-requests",
      storage: createJSONStorage(() => localStorage),
      partialize: (state): AskUserState => ({
        requests: state.requests,
      }),
    },
  ),
);
