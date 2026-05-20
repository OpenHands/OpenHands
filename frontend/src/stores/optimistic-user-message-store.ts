import { create } from "zustand";

interface OptimisticUserMessageState {
  optimisticUserMessage: string | null;
  /**
   * True when the optimistic message has been queued server-side via the
   * REST pending-message endpoint but not yet delivered over the WebSocket.
   * The chat surface uses this to render a "Delivering..." status indicator.
   */
  isPendingDelivery: boolean;
}

interface OptimisticUserMessageActions {
  setOptimisticUserMessage: (
    message: string,
    isPendingDelivery?: boolean,
  ) => void;
  getOptimisticUserMessage: () => string | null;
  isOptimisticUserMessagePending: () => boolean;
  removeOptimisticUserMessage: () => void;
}

type OptimisticUserMessageStore = OptimisticUserMessageState &
  OptimisticUserMessageActions;

const initialState: OptimisticUserMessageState = {
  optimisticUserMessage: null,
  isPendingDelivery: false,
};

export const useOptimisticUserMessageStore = create<OptimisticUserMessageStore>(
  (set, get) => ({
    ...initialState,

    setOptimisticUserMessage: (
      message: string,
      isPendingDelivery: boolean = false,
    ) =>
      set(() => ({
        optimisticUserMessage: message,
        isPendingDelivery,
      })),

    getOptimisticUserMessage: () => get().optimisticUserMessage,

    isOptimisticUserMessagePending: () => get().isPendingDelivery,

    removeOptimisticUserMessage: () =>
      set(() => ({
        optimisticUserMessage: null,
        isPendingDelivery: false,
      })),
  }),
);
