import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

export interface SecretRequest {
  conversationId: string;
  toolCallId: string;
  name: string;
  reason: string | null;
}

interface SecretRequestState {
  /** Pending requests, oldest first; each is shown as a card in the chat. */
  requests: SecretRequest[];
}

interface SecretRequestActions {
  enqueue: (request: SecretRequest) => void;
  dismiss: (toolCallId: string) => void;
}

// Persisted so a reload reopens a pending request: its tool call is already
// claimed, so the replayed ActionEvent will not queue it again.
export const useSecretRequestStore = create<
  SecretRequestState & SecretRequestActions
>()(
  persist(
    (set) => ({
      requests: [],
      enqueue: (request) =>
        set((s) => ({ requests: [...s.requests, request] })),
      dismiss: (toolCallId) =>
        set((s) => ({
          requests: s.requests.filter((r) => r.toolCallId !== toolCallId),
        })),
    }),
    {
      name: "openhands-pending-secret-requests",
      storage: createJSONStorage(() => localStorage),
      partialize: (state): SecretRequestState => ({
        requests: state.requests,
      }),
    },
  ),
);
