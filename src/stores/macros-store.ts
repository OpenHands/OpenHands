import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import {
  MACROS_STORAGE_KEY,
  MACRO_CATALOG,
  createDefaultMacros,
  createMacroId,
  isMacro,
  type Macro,
} from "#/utils/macros";

interface MacrosState {
  macros: Macro[];
}

interface MacrosActions {
  addMacro: (input: { title: string; prompt: string }) => void;
  updateMacro: (id: string, patch: { title: string; prompt: string }) => void;
  removeMacro: (id: string) => void;
  addFromCatalog: (catalogId: string) => void;
}

type MacrosStore = MacrosState & MacrosActions;

const initialState: MacrosState = {
  macros: createDefaultMacros(),
};

export const useMacrosStore = create<MacrosStore>()(
  persist(
    (set) => ({
      ...initialState,

      addMacro: ({ title, prompt }) =>
        set((state) => ({
          macros: [
            ...state.macros,
            {
              id: createMacroId(),
              title,
              prompt,
            },
          ],
        })),

      updateMacro: (id, patch) =>
        set((state) => ({
          macros: state.macros.map((macro) =>
            macro.id === id ? { ...macro, ...patch } : macro,
          ),
        })),

      removeMacro: (id) =>
        set((state) => ({
          macros: state.macros.filter((macro) => macro.id !== id),
        })),

      addFromCatalog: (catalogId) =>
        set((state) => {
          if (state.macros.some((macro) => macro.catalogId === catalogId)) {
            return state;
          }
          const entry = MACRO_CATALOG.find((item) => item.id === catalogId);
          if (!entry) {
            return state;
          }
          return {
            macros: [
              ...state.macros,
              {
                id: entry.id,
                title: entry.titleKey,
                prompt: entry.prompt,
                catalogId: entry.id,
              },
            ],
          };
        }),
    }),
    {
      name: MACROS_STORAGE_KEY,
      storage: createJSONStorage(() => localStorage),
      partialize: (state): MacrosState => ({ macros: state.macros }),
      merge: (persisted, current) => {
        const stored = persisted as MacrosState | undefined;
        const macros = Array.isArray(stored?.macros)
          ? stored.macros.filter(isMacro)
          : current.macros;
        return {
          ...current,
          macros:
            stored && Array.isArray(stored.macros) ? macros : current.macros,
        };
      },
    },
  ),
);
