import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

/**
 * Per-profile "remembered" custom ACP model ids, keyed by AgentProfile UUID
 * ({@link AgentProfile.id}). Mirrors {@link usePinnedConversationsStore}
 * (same persist/partialize/prune shape) — see
 * ``src/stores/pinned-conversations-store.ts``.
 *
 * The Agent-profile editor's ACP model picker (`src/routes/agent-settings.tsx`)
 * offers a free-text override alongside the curated + models.dev suggestions
 * (`ACP_CUSTOM_MODEL_KEY`). Historically that override was one-shot: it saved
 * to the profile's ``acp_model`` field but never became a selectable option
 * again. This store remembers it client-side so re-opening the editor (or
 * `useAcpModelChoices`, which reads it) offers it back as a `source: "custom"`
 * choice.
 */
interface AcpCustomModelsState {
  customModelsByProfileId: Record<string, string[]>;
}

interface AcpCustomModelsActions {
  addCustomModel: (profileId: string, modelId: string) => void;
  removeCustomModel: (profileId: string, modelId: string) => void;
  pruneMissingProfiles: (existingProfileIds: readonly string[]) => void;
}

type AcpCustomModelsStore = AcpCustomModelsState & AcpCustomModelsActions;

const initialState: AcpCustomModelsState = {
  customModelsByProfileId: {},
};

function getCustomModelsForProfile(
  customModelsByProfileId: Record<string, string[]>,
  profileId: string,
): string[] {
  return customModelsByProfileId[profileId] ?? [];
}

export const useAcpCustomModelsStore = create<AcpCustomModelsStore>()(
  persist(
    (set, get) => ({
      ...initialState,

      addCustomModel: (profileId, modelId) => {
        const trimmed = modelId.trim();
        if (!trimmed) {
          return;
        }
        const current = getCustomModelsForProfile(
          get().customModelsByProfileId,
          profileId,
        );
        if (current.includes(trimmed)) {
          return;
        }
        set((state) => ({
          customModelsByProfileId: {
            ...state.customModelsByProfileId,
            [profileId]: [trimmed, ...current],
          },
        }));
      },

      removeCustomModel: (profileId, modelId) => {
        const current = getCustomModelsForProfile(
          get().customModelsByProfileId,
          profileId,
        );
        if (!current.includes(modelId)) {
          return;
        }
        set((state) => ({
          customModelsByProfileId: {
            ...state.customModelsByProfileId,
            [profileId]: current.filter((id) => id !== modelId),
          },
        }));
      },

      pruneMissingProfiles: (existingProfileIds) => {
        const existing = new Set(existingProfileIds);
        const current = get().customModelsByProfileId;
        const entries = Object.entries(current);
        const pruned = entries.filter(([profileId]) => existing.has(profileId));
        if (pruned.length === entries.length) {
          return;
        }
        set({ customModelsByProfileId: Object.fromEntries(pruned) });
      },
    }),
    {
      name: "acp-custom-models",
      storage: createJSONStorage(() => localStorage),
      partialize: (state): AcpCustomModelsState => ({
        customModelsByProfileId: state.customModelsByProfileId,
      }),
    },
  ),
);
