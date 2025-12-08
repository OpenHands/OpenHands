import { create } from "zustand";
import { V1ExecutionStatus } from "#/types/v1/core/base/common";

interface V1ConversationStateStore {
  execution_status: V1ExecutionStatus | null;

  /**
   * Set the agent status
   */
  setExecutionStatus: (execution_status: V1ExecutionStatus) => void;

  /**
   * Reset the store to initial state
   */
  reset: () => void;
}

export const useV1ConversationStateStore = create<V1ConversationStateStore>(
  (set) => ({
    execution_status: null,

    setExecutionStatus: (execution_status: V1ExecutionStatus) =>
      set({ execution_status }),

    reset: () => set({ execution_status: null }),
  }),
);
