import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

/**
 * Ids of cloud conversation start tasks this browser launched that have not
 * settled yet.
 *
 * The cloud app-server exposes start tasks only through the batch
 * `/api/v1/app-conversations/start-tasks?ids=…` lookup — there is no "list my
 * in-progress tasks" endpoint — so the ids have to be remembered client-side.
 * `useCreateConversation` tracks a task as soon as it comes back without an
 * `app_conversation_id`, `useStartTasks` batch-fetches the tracked ids for the
 * sidebar, and each id is untracked once its task reaches READY or ERROR (or
 * the backend stops returning it).
 *
 * Persisted so reloading while a sandbox is still provisioning keeps the
 * in-progress card, and scoped by backend id because a task id only resolves
 * on the cloud backend that issued it.
 */
interface InFlightStartTasksState {
  taskIdsByBackendId: Record<string, string[]>;
}

interface InFlightStartTasksActions {
  trackStartTask: (backendId: string, taskId: string) => void;
  untrackStartTasks: (backendId: string, taskIds: readonly string[]) => void;
}

type InFlightStartTasksStore = InFlightStartTasksState &
  InFlightStartTasksActions;

/**
 * Upper bound on remembered ids per backend. A task whose id is never pruned
 * (the tab closed mid-flight and the backend has since forgotten the task)
 * must not grow the list without end.
 */
const MAX_TRACKED_START_TASKS = 20;

const initialState: InFlightStartTasksState = {
  taskIdsByBackendId: {},
};

export const useInFlightStartTasksStore = create<InFlightStartTasksStore>()(
  persist(
    (set, get) => ({
      ...initialState,

      trackStartTask: (backendId, taskId) => {
        const current = get().taskIdsByBackendId[backendId] ?? [];
        if (current.includes(taskId)) {
          return;
        }
        set((state) => ({
          taskIdsByBackendId: {
            ...state.taskIdsByBackendId,
            [backendId]: [taskId, ...current].slice(0, MAX_TRACKED_START_TASKS),
          },
        }));
      },

      untrackStartTasks: (backendId, taskIds) => {
        const settled = new Set(taskIds);
        const current = get().taskIdsByBackendId[backendId] ?? [];
        const remaining = current.filter((id) => !settled.has(id));
        if (remaining.length === current.length) {
          return;
        }
        set((state) => ({
          taskIdsByBackendId: {
            ...state.taskIdsByBackendId,
            [backendId]: remaining,
          },
        }));
      },
    }),
    {
      name: "in-flight-start-tasks",
      storage: createJSONStorage(() => localStorage),
      partialize: (state): InFlightStartTasksState => ({
        taskIdsByBackendId: state.taskIdsByBackendId,
      }),
    },
  ),
);

/** Non-React entry point for the create-conversation launch path. */
export function trackStartTask(backendId: string, taskId: string): void {
  useInFlightStartTasksStore.getState().trackStartTask(backendId, taskId);
}
