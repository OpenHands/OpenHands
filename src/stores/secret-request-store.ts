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
  name: string;
  description: string;
  value: string;
  showValue: boolean;
}

/**
 * Mutation key scoping an in-flight secret save to one request's card. The card
 * reads its pending state with `useIsMutating`, which lives on the mutation
 * cache entry and survives the card unmounting across the virtualization
 * threshold; a per-form `isPending` would read idle on the remount.
 */
export const secretRequestMutationKey = (toolCallId: string): string[] => [
  "secret-request",
  toolCallId,
];

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
      dismiss: (toolCallId) => {
        // The draft is in-memory only and belongs to the request, so dropping
        // the request drops what the user typed with it.
        useSecretRequestDraftStore.getState().clearDraft(toolCallId);
        set((s) => ({
          requests: s.requests.filter((r) => r.toolCallId !== toolCallId),
        }));
      },
    }),
    {
      name: "openhands-pending-secret-requests",
      storage: createJSONStorage(() => localStorage),
    },
  ),
);

interface SecretRequestDraftState {
  /**
   * Typed fields per tool call (name, description, value, show toggle). A card
   * can be unmounted and remounted when the chat crosses the virtualization
   * threshold (its row moves to the trailing list), which would otherwise lose
   * what the user is typing while the agent keeps working.
   *
   * A separate store, deliberately not persisted: the value is a secret, so it
   * must never reach `localStorage`, and keeping it out of the persisted store
   * also means a keystroke does not re-serialize and rewrite the request list.
   */
  drafts: Record<string, SecretRequestDraft>;
  setDraft: (toolCallId: string, draft: SecretRequestDraft) => void;
  clearDraft: (toolCallId: string) => void;
}

export const useSecretRequestDraftStore = create<SecretRequestDraftState>(
  (set) => ({
    drafts: {},
    setDraft: (toolCallId, draft) =>
      set((s) => ({ drafts: { ...s.drafts, [toolCallId]: draft } })),
    clearDraft: (toolCallId) =>
      set((s) => {
        const { [toolCallId]: _cleared, ...drafts } = s.drafts;
        return { drafts };
      }),
  }),
);
