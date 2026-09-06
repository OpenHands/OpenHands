import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

const STORAGE_KEY = "subscription-model-preferences";

interface SubscriptionModelPreferencesState {
  /** Toggle keys (`source:nativeId`) the user turned off. Missing = enabled. */
  disabledKeys: string[];
  setEnabled: (key: string, enabled: boolean) => void;
}

export const useSubscriptionModelPreferencesStore =
  create<SubscriptionModelPreferencesState>()(
    persist(
      (set, get) => ({
        disabledKeys: [],
        setEnabled: (key, enabled) => {
          const disabled = new Set(get().disabledKeys);
          if (enabled) disabled.delete(key);
          else disabled.add(key);
          set({ disabledKeys: [...disabled].sort() });
        },
      }),
      {
        name: STORAGE_KEY,
        storage: createJSONStorage(() => localStorage),
        partialize: (state) => ({ disabledKeys: state.disabledKeys }),
      },
    ),
  );

export function isSubscriptionModelEnabled(
  disabledKeys: string[],
  key: string,
): boolean {
  return !disabledKeys.includes(key);
}
