import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

export interface SecretRequest {
  conversationId: string;
  toolCallId: string;
  name: string;
  reason: string | null;
}

/** In-progress input for a request's card, held across a card remount. */
export interface SecretRequestDraft {
  value: string;
  showValue: boolean;
}

interface SecretRequestState {
  /** Pending requests, oldest first; each is shown as a card in the chat. */
  requests: SecretRequest[];
  /**
   * Typed value + show toggle per tool call. A card can be unmounted and
   * remounted when the chat crosses the virtualization threshold (its row
   * moves to the trailing list), which would otherwise lose what the user is
   * typing while the agent keeps working. In-memory only: never persisted, so
   * a typed secret value is not written to `localStorage`.
   */
  drafts: Record<string, SecretRequestDraft>;
}

interface SecretRequestActions {
  enqueue: (request: SecretRequest) => void;
  dismiss: (toolCallId: string) => void;
  setDraft: (toolCallId: string, draft: SecretRequestDraft) => void;
}

// Persisted so a reload reopens a pending request: its tool call is already
// claimed, so the replayed ActionEvent will not queue it again.
export const useSecretRequestStore = create<
  SecretRequestState & SecretRequestActions
>()(
  persist(
    (set) => ({
      requests: [],
      drafts: {},
      enqueue: (request) =>
        set((s) => ({ requests: [...s.requests, request] })),
      dismiss: (toolCallId) =>
        set((s) => {
          const { [toolCallId]: _dismissed, ...drafts } = s.drafts;
          return {
            requests: s.requests.filter((r) => r.toolCallId !== toolCallId),
            drafts,
          };
        }),
      setDraft: (toolCallId, draft) =>
        set((s) => ({ drafts: { ...s.drafts, [toolCallId]: draft } })),
    }),
    {
      name: "openhands-pending-secret-requests",
      storage: createJSONStorage(() => localStorage),
      partialize: (state): SecretRequestState => ({
        requests: state.requests,
        drafts: {},
      }),
    },
  ),
);
