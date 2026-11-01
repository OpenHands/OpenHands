import type { PendingTaskAttachments } from "./pending-task-attachments-store";
import { create } from "zustand";

export const CONVERSATION_LAUNCH_PREFIX = "launch-";

export interface ConversationLaunch {
  id: string;
  backendId: string;
  orgId: string | null;
  message: string;
  files: File[];
  attachments?: PendingTaskAttachments;
  targetId?: string;
  error?: string;
}

interface LaunchState {
  launches: Record<string, ConversationLaunch>;
  begin: (launch: Omit<ConversationLaunch, "id">) => string;
  update: (
    id: string,
    update: Partial<Pick<ConversationLaunch, "targetId" | "error">>,
  ) => void;
  remove: (id: string) => void;
}

// In-memory handoff only: a client launch ID is never sent to the server.
export const useConversationLaunchStore = create<LaunchState>((set) => ({
  launches: {},
  begin: (launch) => {
    const id = `${CONVERSATION_LAUNCH_PREFIX}${crypto.randomUUID()}`;
    set((state) => ({
      launches: { ...state.launches, [id]: { ...launch, id } },
    }));
    return id;
  },
  update: (id, update) =>
    set((state) =>
      state.launches[id]
        ? {
            launches: {
              ...state.launches,
              [id]: {
                ...state.launches[id],
                ...update,
                ...(update.targetId
                  ? { files: [], attachments: undefined }
                  : {}),
              },
            },
          }
        : state,
    ),
  remove: (id) =>
    set((state) => {
      const { [id]: _removed, ...launches } = state.launches;
      return { launches };
    }),
}));
