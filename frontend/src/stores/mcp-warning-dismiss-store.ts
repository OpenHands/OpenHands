import { create } from "zustand";
import { devtools } from "zustand/middleware";

function dismissKey(conversationId: string, serverId: string): string {
  return `${conversationId}:${serverId}`;
}

interface McpWarningDismissState {
  dismissedKeys: string[];
}

interface McpWarningDismissActions {
  dismiss: (conversationId: string, serverId: string) => void;
  isDismissed: (conversationId: string, serverId: string) => boolean;
}

type McpWarningDismissStore = McpWarningDismissState & McpWarningDismissActions;

const initialState: McpWarningDismissState = { dismissedKeys: [] };

export const useMcpWarningDismissStore = create<McpWarningDismissStore>()(
  devtools(
    (set, get) => ({
      ...initialState,
      dismiss: (conversationId, serverId) => {
        const key = dismissKey(conversationId, serverId);
        set((state) =>
          state.dismissedKeys.includes(key)
            ? state
            : { dismissedKeys: [...state.dismissedKeys, key] },
        );
      },
      isDismissed: (conversationId, serverId) =>
        get().dismissedKeys.includes(dismissKey(conversationId, serverId)),
    }),
    { name: "McpWarningDismissStore" },
  ),
);
