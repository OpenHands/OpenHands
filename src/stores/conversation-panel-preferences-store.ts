import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import {
  type AutomationFilterMode,
  type ConversationSortField,
  type OrganizeMode,
  type ThreadScope,
} from "#/components/features/conversation-panel/conversation-panel-list-helpers";

/**
 * User-toggleable display preferences for the sidebar conversation list
 * filter menu. These are intentionally persisted to localStorage (via the
 * same `zustand/persist` pattern used by `home-store` and `workspaces-store`)
 * so the menu state survives full reloads.
 *
 * To add a new preference exposed by the filter menu:
 *   1. Add a field here with a sensible default in `initialState`.
 *   2. Add matching `setX`/`toggleX` actions below.
 *   3. Read/write through the store in `conversation-panel.tsx`.
 * No additional plumbing (storage keys, sanitization, etc.) is required —
 * `persist` handles migration of unknown fields gracefully.
 */
interface ConversationPanelPreferencesState {
  showOlderConversations: boolean;
  showArchivedConversations: boolean;
  showRepoBranchMetadata: boolean;
  showLlmProfiles: boolean;
  showTagsMetadata: boolean;
  showHoverMetadata: boolean;
  organizeMode: OrganizeMode;
  conversationSort: ConversationSortField;
  threadScope: ThreadScope;
  automationFilterMode: AutomationFilterMode;
  selectedAutomationNames: string[];
  selectedTagFacets: string[];
  /**
   * Whether the layouts menu exposes the Tag Filters section. Off by
   * default: tag filtering is opt-in surface area, matching the Advanced
   * options "Tag Filters" toggle that gates it.
   */
  tagFiltersEnabled: boolean;
  groupFolderOrder: string[];
}

/**
 * The preference subset a conversation-layouts preset may set. Presets are
 * partial bundles over these fields; any manual deviation in the Advanced
 * options modal drops the active preset to "Custom".
 */
export type LayoutSettingsSlice = Pick<
  ConversationPanelPreferencesState,
  | "organizeMode"
  | "conversationSort"
  | "threadScope"
  | "showOlderConversations"
  | "showRepoBranchMetadata"
  | "showLlmProfiles"
  | "showTagsMetadata"
  | "showHoverMetadata"
>;

interface ConversationPanelPreferencesActions {
  setShowOlderConversations: (value: boolean) => void;
  toggleShowOlderConversations: () => void;
  setShowArchivedConversations: (value: boolean) => void;
  toggleShowArchivedConversations: () => void;
  setShowRepoBranchMetadata: (value: boolean) => void;
  toggleShowRepoBranchMetadata: () => void;
  setShowLlmProfiles: (value: boolean) => void;
  toggleShowLlmProfiles: () => void;
  setShowTagsMetadata: (value: boolean) => void;
  toggleShowTagsMetadata: () => void;
  setShowHoverMetadata: (value: boolean) => void;
  toggleShowHoverMetadata: () => void;
  setOrganizeMode: (value: OrganizeMode) => void;
  setConversationSort: (value: ConversationSortField) => void;
  setThreadScope: (value: ThreadScope) => void;
  setAutomationFilterMode: (value: AutomationFilterMode) => void;
  toggleAutomationName: (name: string) => void;
  /**
   * Bar-facing toggle: selecting a name implies `only-automations` mode;
   * removing the last selected name returns the mode to `all`. The popup's
   * mode rows use `setAutomationFilterMode`, which clears the selection when
   * the mode leaves `only-automations` — the two surfaces cannot disagree.
   */
  toggleAutomationNameAndMode: (name: string) => void;
  /** Clears both filter selections and returns automation mode to `all`. */
  clearFilterSelections: () => void;
  toggleTagFacet: (facet: string) => void;
  setTagFiltersEnabled: (value: boolean) => void;
  toggleTagFiltersEnabled: () => void;
  /** Applies a layout preset's partial bundle in one set(). */
  applyLayoutSettings: (settings: Partial<LayoutSettingsSlice>) => void;
  setGroupFolderOrder: (order: readonly string[]) => void;
}

type ConversationPanelPreferencesStore = ConversationPanelPreferencesState &
  ConversationPanelPreferencesActions;

const initialState: ConversationPanelPreferencesState = {
  showOlderConversations: true,
  showArchivedConversations: false,
  showRepoBranchMetadata: false,
  showLlmProfiles: false,
  showTagsMetadata: false,
  showHoverMetadata: true,
  organizeMode: "chronological",
  conversationSort: "updated",
  threadScope: "all",
  automationFilterMode: "all",
  selectedAutomationNames: [],
  selectedTagFacets: [],
  tagFiltersEnabled: false,
  groupFolderOrder: [],
};

export const useConversationPanelPreferencesStore =
  create<ConversationPanelPreferencesStore>()(
    persist(
      (set) => ({
        ...initialState,

        setShowOlderConversations: (value) =>
          set(() => ({ showOlderConversations: value })),
        toggleShowOlderConversations: () =>
          set((state) => ({
            showOlderConversations: !state.showOlderConversations,
          })),

        setShowArchivedConversations: (value) =>
          set(() => ({ showArchivedConversations: value })),
        toggleShowArchivedConversations: () =>
          set((state) => ({
            showArchivedConversations: !state.showArchivedConversations,
          })),

        setShowRepoBranchMetadata: (value) =>
          set(() => ({ showRepoBranchMetadata: value })),
        toggleShowRepoBranchMetadata: () =>
          set((state) => ({
            showRepoBranchMetadata: !state.showRepoBranchMetadata,
          })),

        setShowLlmProfiles: (value) => set(() => ({ showLlmProfiles: value })),
        toggleShowLlmProfiles: () =>
          set((state) => ({
            showLlmProfiles: !state.showLlmProfiles,
          })),

        setShowTagsMetadata: (value) =>
          set(() => ({ showTagsMetadata: value })),
        toggleShowTagsMetadata: () =>
          set((state) => ({
            showTagsMetadata: !state.showTagsMetadata,
          })),

        setShowHoverMetadata: (value) =>
          set(() => ({ showHoverMetadata: value })),
        toggleShowHoverMetadata: () =>
          set((state) => ({
            showHoverMetadata: !state.showHoverMetadata,
          })),

        setOrganizeMode: (value) => set(() => ({ organizeMode: value })),
        setConversationSort: (value) =>
          set(() => ({ conversationSort: value })),
        setThreadScope: (value) => set(() => ({ threadScope: value })),
        setAutomationFilterMode: (value) =>
          set((state) => ({
            automationFilterMode: value,
            // Name selections only make sense in only-automations mode;
            // leaving the mode clears them (self-healing — a stale selection
            // must never silently narrow an unfiltered-looking list).
            selectedAutomationNames:
              value === "only-automations" ? state.selectedAutomationNames : [],
          })),
        toggleAutomationName: (name) =>
          set((state) => ({
            selectedAutomationNames: state.selectedAutomationNames.includes(
              name,
            )
              ? state.selectedAutomationNames.filter(
                  (existing) => existing !== name,
                )
              : [...state.selectedAutomationNames, name],
          })),
        toggleAutomationNameAndMode: (name) =>
          set((state) => {
            const selectedAutomationNames =
              state.selectedAutomationNames.includes(name)
                ? state.selectedAutomationNames.filter(
                    (existing) => existing !== name,
                  )
                : [...state.selectedAutomationNames, name];
            return {
              selectedAutomationNames,
              automationFilterMode:
                selectedAutomationNames.length > 0 ? "only-automations" : "all",
            };
          }),
        clearFilterSelections: () =>
          set(() => ({
            selectedTagFacets: [],
            selectedAutomationNames: [],
            automationFilterMode: "all",
          })),
        toggleTagFacet: (facet) =>
          set((state) => ({
            selectedTagFacets: state.selectedTagFacets.includes(facet)
              ? state.selectedTagFacets.filter((existing) => existing !== facet)
              : [...state.selectedTagFacets, facet],
          })),
        setGroupFolderOrder: (order) =>
          set(() => ({ groupFolderOrder: [...order] })),

        setTagFiltersEnabled: (value) =>
          set(() => ({ tagFiltersEnabled: value })),
        toggleTagFiltersEnabled: () =>
          set((state) => ({ tagFiltersEnabled: !state.tagFiltersEnabled })),
        applyLayoutSettings: (settings) => set(() => ({ ...settings })),
      }),
      {
        name: "conversation-panel-preferences",
        storage: createJSONStorage(() => localStorage),
        // Only persist the data fields — actions are recreated on each load.
        partialize: (state): ConversationPanelPreferencesState => ({
          showOlderConversations: state.showOlderConversations,
          showArchivedConversations: state.showArchivedConversations,
          showRepoBranchMetadata: state.showRepoBranchMetadata,
          showLlmProfiles: state.showLlmProfiles,
          showTagsMetadata: state.showTagsMetadata,
          showHoverMetadata: state.showHoverMetadata,
          organizeMode: state.organizeMode,
          conversationSort: state.conversationSort,
          threadScope: state.threadScope,
          automationFilterMode: state.automationFilterMode,
          selectedAutomationNames: state.selectedAutomationNames,
          selectedTagFacets: state.selectedTagFacets,
          tagFiltersEnabled: state.tagFiltersEnabled,
          groupFolderOrder: state.groupFolderOrder,
        }),
      },
    ),
  );
