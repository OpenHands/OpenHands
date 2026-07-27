import { create } from "zustand";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

interface ConversationStateStore {
  execution_status: ExecutionStatus | null;

  /**
   * Set the agent status
   */
  setExecutionStatus: (execution_status: ExecutionStatus) => void;

  /**
   * Reset the store to initial state
   */
  reset: () => void;
}

export const useConversationStateStore = create<ConversationStateStore>(
  (set) => ({
    execution_status: null,

    setExecutionStatus: (execution_status: ExecutionStatus) =>
      set({ execution_status }),

    reset: () => set({ execution_status: null }),
  }),
);
