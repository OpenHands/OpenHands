import { create } from "zustand";
import { persist } from "zustand/middleware";

interface SandboxOOMState {
  /** Event IDs that the user has dismissed. */
  dismissedEventIds: Record<string, true>;
}

interface SandboxOOMActions {
  dismiss: (eventIds: string[]) => void;
  clearAll: () => void;
}

export type SandboxOOMStore = SandboxOOMState & SandboxOOMActions;

const STORAGE_KEY = "openhands-sandbox-oom-dismissals";

export const useSandboxOOMStore = create<SandboxOOMStore>()(
  persist(
    (set) => ({
      dismissedEventIds: {},
      dismiss: (eventIds) =>
        set((state) => ({
          dismissedEventIds: {
            ...state.dismissedEventIds,
            ...Object.fromEntries(eventIds.map((id) => [id, true as const])),
          },
        })),
      clearAll: () => set({ dismissedEventIds: {} }),
    }),
    {
      name: STORAGE_KEY,
    },
  ),
);
