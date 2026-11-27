import { create } from "zustand";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

interface ConversationStateStore {
  /**
   * Latest execution status per conversation, fed by the main and planning
   * WebSocket handlers. Scoped by conversation id so the planning helper
   * conversation's own run/idle transitions can never overwrite the main
   * conversation's status (or vice versa) — see conversation-websocket-context.tsx.
   */
  executionStatusByConversation: Record<string, ExecutionStatus>;

  /**
   * Set the agent status for a specific conversation.
   */
  setExecutionStatus: (
    conversationId: string,
    execution_status: ExecutionStatus,
  ) => void;

  /**
   * Reset the store to initial state
   */
  reset: () => void;
}

export const useConversationStateStore = create<ConversationStateStore>(
  (set) => ({
    executionStatusByConversation: {},

    setExecutionStatus: (conversationId, execution_status) =>
      set((state) => ({
        executionStatusByConversation: {
          ...state.executionStatusByConversation,
          [conversationId]: execution_status,
        },
      })),

    reset: () => set({ executionStatusByConversation: {} }),
  }),
);
