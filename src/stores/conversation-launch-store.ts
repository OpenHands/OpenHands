import {
  sessionActivityKey,
  useSessionActivityStore,
} from "./session-activity-store";
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
  conversationId?: string;
  listed?: boolean;
  error?: string;
}

interface LaunchState {
  launches: Record<string, ConversationLaunch>;
  begin: (launch: Omit<ConversationLaunch, "id">) => string;
  update: (
    id: string,
    update: Partial<
      Pick<
        ConversationLaunch,
        "targetId" | "error" | "conversationId" | "listed"
      >
    >,
  ) => void;
  remove: (id: string) => void;
}

// In-memory handoff only: a client launch ID is never sent to the server.
export const useConversationLaunchStore = create<LaunchState>((set, get) => ({
  launches: {},
  begin: (launch) => {
    const id = `${CONVERSATION_LAUNCH_PREFIX}${crypto.randomUUID()}`;
    useSessionActivityStore
      .getState()
      .begin(
        sessionActivityKey(launch.backendId, launch.orgId, id),
        "initialize",
      );
    set((state) => ({
      launches: { ...state.launches, [id]: { ...launch, id } },
    }));
    return id;
  },
  update: (id, update) => {
    const launch = get().launches[id];
    if (launch && "error" in update) {
      useSessionActivityStore
        .getState()
        .observe(
          sessionActivityKey(launch.backendId, launch.orgId, id),
          update.error ? "ERROR" : "WORKING",
        );
    }
    if (launch && update.targetId) {
      const activity = useSessionActivityStore.getState();
      const from = sessionActivityKey(launch.backendId, launch.orgId, id);
      const to = sessionActivityKey(
        launch.backendId,
        launch.orgId,
        update.targetId,
      );
      activity.transfer(from, to);
      if (!update.targetId.startsWith("task-")) activity.observe(to, "READY");
    }
    set((state) =>
      state.launches[id] &&
      Object.entries(update).some(
        ([key, value]) =>
          state.launches[id][key as keyof ConversationLaunch] !== value,
      )
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
    );
  },
  remove: (id) =>
    set((state) => {
      const { [id]: _removed, ...launches } = state.launches;
      return { launches };
    }),
}));
