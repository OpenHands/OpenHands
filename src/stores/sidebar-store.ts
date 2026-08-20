import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

interface SidebarState {
  collapsed: boolean;
}

interface SidebarActions {
  setCollapsed: (next: boolean | ((prev: boolean) => boolean)) => void;
  toggleCollapsed: () => void;
}

type SidebarStore = SidebarState & SidebarActions;

const STORAGE_KEY = "openhands-sidebar";
const LEGACY_STORAGE_KEY = "openhands-sidebar-collapsed";

export const useSidebarStore = create<SidebarStore>()(
  persist(
    (set) => ({
      collapsed: false,
      setCollapsed: (next) =>
        set((state) => ({
          collapsed: typeof next === "function" ? next(state.collapsed) : next,
        })),
      toggleCollapsed: () => set((state) => ({ collapsed: !state.collapsed })),
    }),
    {
      name: STORAGE_KEY,
      storage: createJSONStorage(() => localStorage),
      partialize: (state): SidebarState => ({ collapsed: state.collapsed }),
    },
  ),
);

// One-shot migration from the previous raw-string format
// (`openhands-sidebar-collapsed` = `"true"`/`"false"`). Runs once at import
// time; only seeds the store when the new key is absent so it never clobbers
// a user choice made after the upgrade.
if (typeof window !== "undefined") {
  try {
    const legacy = window.localStorage.getItem(LEGACY_STORAGE_KEY);
    const upgraded = window.localStorage.getItem(STORAGE_KEY) !== null;
    if (!upgraded && (legacy === "true" || legacy === "false")) {
      useSidebarStore.setState({ collapsed: legacy === "true" });
    }
    if (legacy !== null) {
      window.localStorage.removeItem(LEGACY_STORAGE_KEY);
    }
  } catch {
    /* localStorage unavailable; the in-memory default is fine */
  }
}
