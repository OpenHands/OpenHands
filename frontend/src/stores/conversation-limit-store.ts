import { create } from "zustand";

interface ConversationLimitState {
  isOpen: boolean;
  limit: number | null;
}

interface ConversationLimitActions {
  showLimitModal: (limit: number) => void;
  closeLimitModal: () => void;
}

type ConversationLimitStore = ConversationLimitState & ConversationLimitActions;

const initialState: ConversationLimitState = {
  isOpen: false,
  limit: null,
};

export const useConversationLimitStore = create<ConversationLimitStore>(
  (set) => ({
    ...initialState,

    showLimitModal: (limit: number) =>
      set(() => ({
        isOpen: true,
        limit,
      })),

    closeLimitModal: () =>
      set(() => ({
        isOpen: false,
        limit: null,
      })),
  }),
);
