import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import {
  DEFAULT_OLDER_CONVERSATION_CUTOFF,
  isOlderConversationCutoff,
  type AutomationFilterMode,
  type ConversationSortField,
  type OlderConversationCutoff,
  type OrganizeMode,
  type ThreadScope,
} from "#/components/features/conversation-panel/conversation-panel-list-helpers";

/**
 * User-toggleable display preferences for the sidebar conversation list
 * (the layouts menu and the advanced-options modal behind it). These are
 * intentionally persisted to localStorage (via the same `zustand/persist`
 * pattern used by `home-store` and `workspaces-store`) so the menu state
 * survives full reloads.
 *
 * Every persisted narrowing field must have a control that can see and undo
 * it, or a reload silently hides conversations with nothing on screen to say
 * why — see `ConversationActiveTagFilters`.
 *
 * To add a new preference exposed by those menus:
 *   1. Add a field here with a sensible default in `initialState`.
 *   2. Add matching `setX`/`toggleX` actions below.
 *   3. Read/write through the store in `conversation-panel.tsx`.
 * No additional plumbing (storage keys, sanitization, etc.) is required —
 * `persist` handles migration of unknown fields gracefully.
 */
interface ConversationPanelPreferencesState {
  showOlderConversations: boolean;
  /**
   * Age threshold used when `showOlderConversations` is false. Conversations
   * last updated before this interval are hidden.
   */
  olderConversationCutoff: OlderConversationCutoff;
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
  groupFolderOrder: string[];
}

/** The complete preference bundle controlled by conversation layout presets. */
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

export const DEFAULT_LAYOUT_SETTINGS: LayoutSettingsSlice = {
  organizeMode: "chronological",
  conversationSort: "updated",
  threadScope: "all",
  showOlderConversations: true,
  showRepoBranchMetadata: false,
  showLlmProfiles: false,
  showTagsMetadata: true,
  showHoverMetadata: true,
};

interface ConversationPanelPreferencesActions {
  setShowOlderConversations: (value: boolean) => void;
  toggleShowOlderConversations: () => void;
  setOlderConversationCutoff: (value: OlderConversationCutoff) => void;
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
   * Clears both facet selections (tags and automation names) — the two
   * narrowings the active-filter strip renders as chips.
   *
   * Deliberately leaves `automationFilterMode` alone: the strip shows no chip
   * for the mode, and it must not silently switch a surface it doesn't show.
   * The mode keeps its own rows in the advanced-options modal.
   */
  clearFilterSelections: () => void;
  toggleTagFacet: (facet: string) => void;
  /** Applies a layout preset's partial bundle in one set(). */
  applyLayoutSettings: (settings: Partial<LayoutSettingsSlice>) => void;
  setGroupFolderOrder: (order: readonly string[]) => void;
}

type ConversationPanelPreferencesStore = ConversationPanelPreferencesState &
  ConversationPanelPreferencesActions;

const initialState: ConversationPanelPreferencesState = {
  ...DEFAULT_LAYOUT_SETTINGS,
  olderConversationCutoff: DEFAULT_OLDER_CONVERSATION_CUTOFF,
  showArchivedConversations: false,
  automationFilterMode: "all",
  selectedAutomationNames: [],
  selectedTagFacets: [],
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
        setOlderConversationCutoff: (value) =>
          set(() => ({
            olderConversationCutoff: isOlderConversationCutoff(value)
              ? value
              : DEFAULT_OLDER_CONVERSATION_CUTOFF,
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
        clearFilterSelections: () =>
          set(() => ({
            selectedTagFacets: [],
            selectedAutomationNames: [],
          })),
        toggleTagFacet: (facet) =>
          set((state) => ({
            selectedTagFacets: state.selectedTagFacets.includes(facet)
              ? state.selectedTagFacets.filter((existing) => existing !== facet)
              : [...state.selectedTagFacets, facet],
          })),
        setGroupFolderOrder: (order) =>
          set(() => ({ groupFolderOrder: [...order] })),

        applyLayoutSettings: (settings) => set(() => ({ ...settings })),
      }),
      {
        name: "conversation-panel-preferences",
        storage: createJSONStorage(() => localStorage),
        // Only persist the data fields — actions are recreated on each load.
        partialize: (state): ConversationPanelPreferencesState => ({
          showOlderConversations: state.showOlderConversations,
          olderConversationCutoff: isOlderConversationCutoff(
            state.olderConversationCutoff,
          )
            ? state.olderConversationCutoff
            : DEFAULT_OLDER_CONVERSATION_CUTOFF,
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
          groupFolderOrder: state.groupFolderOrder,
        }),
      },
    ),
  );
