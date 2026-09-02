import { beforeEach, describe, expect, it } from "vitest";
import { useConversationPanelPreferencesStore } from "#/stores/conversation-panel-preferences-store";

const STORAGE_KEY = "conversation-panel-preferences";

describe("conversation-panel-preferences store", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("defaults to showing older conversations, chronological list (both grouping toggles off), and expected toggles", () => {
    const state = useConversationPanelPreferencesStore.getState();
    expect(state.showOlderConversations).toBe(true);
    expect(state.showRepoBranchMetadata).toBe(false);
    expect(state.showLlmProfiles).toBe(false);
    expect(state.showTagsMetadata).toBe(false);
    expect(state.groupByContainer).toBe(false);
    expect(state.groupByWorkspace).toBe(false);
    expect(state.conversationSort).toBe("updated");
    expect(state.threadScope).toBe("all");
    expect(state.automationFilterMode).toBe("all");
    expect(state.selectedAutomationNames).toEqual([]);
  });

  it("toggles showOlderConversations and persists the new value to localStorage", () => {
    useConversationPanelPreferencesStore
      .getState()
      .toggleShowOlderConversations();

    expect(
      useConversationPanelPreferencesStore.getState().showOlderConversations,
    ).toBe(false);

    const persisted = JSON.parse(
      window.localStorage.getItem(STORAGE_KEY) ?? "{}",
    );
    expect(persisted.state.showOlderConversations).toBe(false);
  });

  it("toggles showRepoBranchMetadata and persists the new value to localStorage", () => {
    useConversationPanelPreferencesStore
      .getState()
      .toggleShowRepoBranchMetadata();

    expect(
      useConversationPanelPreferencesStore.getState().showRepoBranchMetadata,
    ).toBe(true);

    const persisted = JSON.parse(
      window.localStorage.getItem(STORAGE_KEY) ?? "{}",
    );
    expect(persisted.state.showRepoBranchMetadata).toBe(true);
  });

  it("supports explicit setters for both preferences", () => {
    useConversationPanelPreferencesStore
      .getState()
      .setShowOlderConversations(false);
    useConversationPanelPreferencesStore
      .getState()
      .setShowRepoBranchMetadata(true);

    const state = useConversationPanelPreferencesStore.getState();
    expect(state.showOlderConversations).toBe(false);
    expect(state.showRepoBranchMetadata).toBe(true);
  });

  it("persists data fields but not action functions", () => {
    useConversationPanelPreferencesStore
      .getState()
      .toggleShowOlderConversations();

    const persisted = JSON.parse(
      window.localStorage.getItem(STORAGE_KEY) ?? "{}",
    );
    expect(Object.keys(persisted.state).sort()).toEqual([
      "automationFilterMode",
      "conversationSort",
      "groupByContainer",
      "groupByWorkspace",
      "groupFolderOrder",
      "selectedAutomationNames",
      "showArchivedConversations",
      "showHoverMetadata",
      "showLlmProfiles",
      "showOlderConversations",
      "showRepoBranchMetadata",
      "showTagsMetadata",
      "threadScope",
    ]);
  });

  it("exposes setters and a toggler for the LLM-profiles preference", () => {
    useConversationPanelPreferencesStore.getState().setShowLlmProfiles(true);
    expect(
      useConversationPanelPreferencesStore.getState().showLlmProfiles,
    ).toBe(true);

    useConversationPanelPreferencesStore.getState().toggleShowLlmProfiles();
    expect(
      useConversationPanelPreferencesStore.getState().showLlmProfiles,
    ).toBe(false);
  });

  it("updates the grouping toggles, sort, and thread-scope preferences via their setters", () => {
    const store = useConversationPanelPreferencesStore.getState();
    store.setGroupByWorkspace(true);
    store.setConversationSort("created");
    store.setThreadScope("relevant");

    const next = useConversationPanelPreferencesStore.getState();
    expect({
      groupByContainer: next.groupByContainer,
      groupByWorkspace: next.groupByWorkspace,
      conversationSort: next.conversationSort,
      threadScope: next.threadScope,
    }).toEqual({
      // groupByContainer is untouched by setGroupByWorkspace — the two
      // toggles are independent (#15607), not a mutually exclusive mode.
      groupByContainer: false,
      groupByWorkspace: true,
      conversationSort: "created",
      threadScope: "relevant",
    });

    // toggleGroupByContainer flips only its own field, leaving
    // groupByWorkspace (just set above) untouched.
    store.toggleGroupByContainer();
    expect({
      groupByContainer:
        useConversationPanelPreferencesStore.getState().groupByContainer,
      groupByWorkspace:
        useConversationPanelPreferencesStore.getState().groupByWorkspace,
    }).toEqual({ groupByContainer: true, groupByWorkspace: true });

    // Restore defaults so later tests in this file see a pristine store.
    useConversationPanelPreferencesStore.setState({
      groupByContainer: false,
      groupByWorkspace: false,
    });
  });

  it("updates the automation filter mode and toggles selected names via their actions", () => {
    const store = useConversationPanelPreferencesStore.getState();
    store.setAutomationFilterMode("only-automations");
    store.toggleAutomationName("Nightly Audit");
    store.toggleAutomationName("PR Review Bot");
    store.toggleAutomationName("Nightly Audit");

    const next = useConversationPanelPreferencesStore.getState();
    expect({
      automationFilterMode: next.automationFilterMode,
      selectedAutomationNames: next.selectedAutomationNames,
    }).toEqual({
      automationFilterMode: "only-automations",
      // Toggling twice removes the name again; the other selection stays.
      selectedAutomationNames: ["PR Review Bot"],
    });

    // Restore defaults so later tests in this file see a pristine store.
    useConversationPanelPreferencesStore.setState({
      automationFilterMode: "all",
      selectedAutomationNames: [],
    });
  });

  it("rehydrates legacy localStorage payloads (older fields preserved, new fields filled with defaults)", async () => {
    // Simulate a user upgrading from a build that only persisted the two
    // original preferences. After rehydration the store should keep the
    // user's existing choices and fill the new fields from `initialState`.
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        state: {
          showOlderConversations: false,
          showRepoBranchMetadata: true,
        },
        version: 0,
      }),
    );

    await useConversationPanelPreferencesStore.persist.rehydrate();

    const state = useConversationPanelPreferencesStore.getState();
    expect({
      showOlderConversations: state.showOlderConversations,
      showRepoBranchMetadata: state.showRepoBranchMetadata,
      showLlmProfiles: state.showLlmProfiles,
      groupByContainer: state.groupByContainer,
      groupByWorkspace: state.groupByWorkspace,
      conversationSort: state.conversationSort,
      threadScope: state.threadScope,
    }).toEqual({
      // Preserved from the legacy payload.
      showOlderConversations: false,
      showRepoBranchMetadata: true,
      // Filled with defaults for missing fields — including a payload from
      // before #15607 that still carries the old `organizeMode` key: it's
      // simply ignored (an unknown field to `persist`) rather than migrated.
      showLlmProfiles: false,
      groupByContainer: false,
      groupByWorkspace: false,
      conversationSort: "updated",
      threadScope: "all",
    });
  });

  it("does not migrate a pre-#15607 organizeMode payload — grouping resets to off", async () => {
    // Known, deliberate limitation: a user who had the old single "grouped"
    // mode enabled sees a flat/chronological list once after upgrading,
    // rather than automatically mapping to groupByWorkspace=true. Documented
    // here so a future change to add that migration doesn't silently
    // contradict this test.
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        state: { organizeMode: "grouped" },
        version: 0,
      }),
    );

    await useConversationPanelPreferencesStore.persist.rehydrate();

    const state = useConversationPanelPreferencesStore.getState();
    expect(state.groupByContainer).toBe(false);
    expect(state.groupByWorkspace).toBe(false);
  });

  it("preserves an explicitly enabled LLM-profiles preference from persisted storage", async () => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        state: {
          showOlderConversations: true,
          showRepoBranchMetadata: false,
          showLlmProfiles: true,
        },
        version: 0,
      }),
    );

    await useConversationPanelPreferencesStore.persist.rehydrate();

    expect(
      useConversationPanelPreferencesStore.getState().showLlmProfiles,
    ).toBe(true);
  });
});
