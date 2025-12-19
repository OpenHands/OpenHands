import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

interface LlmSettingsViewState {
  view: "basic" | "advanced" | null; // null means no preference stored yet
  setView: (view: "basic" | "advanced") => void;
  clearView: () => void; // Clear stored preference to allow auto-determination
}

export const useLlmSettingsViewStore = create<LlmSettingsViewState>()(
  persist(
    (set) => ({
      view: null, // null = no stored preference, will auto-determine
      setView: (view) => set({ view }),
      clearView: () => set({ view: null }),
    }),
    {
      name: "llm-settings-view-store", // unique name for localStorage
      storage: createJSONStorage(() => localStorage),
    },
  ),
);
