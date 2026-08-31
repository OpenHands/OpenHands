import { create } from "zustand";
import type { FreeModelSet } from "#/utils/format-model-name";

interface FreeModelsState {
  /**
   * DB-driven set of free ``provider/model`` ids (the same channel that carries
   * `verified`). Empty on backends without free metadata (e.g. the local
   * agent-server), so consumers uniformly treat "not in set" as paid.
   */
  freeModels: FreeModelSet;
  /** DB-driven default OpenHands model id (``openhands/<model>``), or null. */
  defaultModel: string | null;
  /** True once the backend flags query has settled at least once. */
  defaultModelReady: boolean;
}

interface FreeModelsActions {
  /**
   * Replaces the DB-driven flags. Called once by the app-level hydrator
   * ({@link useHydrateFreeModels}) after the backend model list is fetched, so
   * leaf components can read the flags synchronously via zustand selectors
   * without needing a QueryClientProvider in scope.
   */
  setFlags: (flags: {
    freeModels: FreeModelSet;
    defaultModel: string | null;
  }) => void;
  markDefaultModelReady: () => void;
}

type FreeModelsStore = FreeModelsState & FreeModelsActions;

const EMPTY_FREE_MODELS: FreeModelSet = new Set<string>();

export const useFreeModelsStore = create<FreeModelsStore>()((set) => ({
  freeModels: EMPTY_FREE_MODELS,
  defaultModel: null,
  defaultModelReady: false,
  setFlags: (flags) =>
    set({
      freeModels: flags.freeModels,
      defaultModel: flags.defaultModel,
      defaultModelReady: true,
    }),
  markDefaultModelReady: () => set({ defaultModelReady: true }),
}));
