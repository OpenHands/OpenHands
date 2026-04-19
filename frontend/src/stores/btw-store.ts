import { create } from "zustand";
import { devtools } from "zustand/middleware";

export type BtwStatus = "pending" | "done" | "error";

export interface BtwEntry {
  id: string;
  question: string;
  response?: string;
  status: BtwStatus;
}

interface BtwState {
  entriesByConversation: Record<string, BtwEntry[]>;
}

interface BtwActions {
  addPending: (conversationId: string, question: string) => string;
  resolve: (conversationId: string, id: string, response: string) => void;
  fail: (conversationId: string, id: string, error: string) => void;
  dismiss: (conversationId: string, id: string) => void;
}

type BtwStore = BtwState & BtwActions;

const initialState: BtwState = { entriesByConversation: {} };

const updateEntries = (
  state: BtwState,
  conversationId: string,
  updater: (entries: BtwEntry[]) => BtwEntry[],
): Pick<BtwState, "entriesByConversation"> => ({
  entriesByConversation: {
    ...state.entriesByConversation,
    [conversationId]: updater(
      state.entriesByConversation[conversationId] ?? [],
    ),
  },
});

export const useBtwStore = create<BtwStore>()(
  devtools(
    (set) => ({
      ...initialState,
      addPending: (conversationId, question) => {
        const id = crypto.randomUUID();
        set((s) =>
          updateEntries(s, conversationId, (entries) => [
            ...entries,
            { id, question, status: "pending" },
          ]),
        );
        return id;
      },
      resolve: (conversationId, id, response) =>
        set((s) =>
          updateEntries(s, conversationId, (entries) =>
            entries.map((e) =>
              e.id === id ? { ...e, response, status: "done" } : e,
            ),
          ),
        ),
      fail: (conversationId, id, error) =>
        set((s) =>
          updateEntries(s, conversationId, (entries) =>
            entries.map((e) =>
              e.id === id ? { ...e, response: error, status: "error" } : e,
            ),
          ),
        ),
      dismiss: (conversationId, id) =>
        set((s) =>
          updateEntries(s, conversationId, (entries) =>
            entries.filter((e) => e.id !== id),
          ),
        ),
    }),
    { name: "BtwStore" },
  ),
);
