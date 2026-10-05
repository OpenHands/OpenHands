import { beforeEach, describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { ConversationLayoutsMenu } from "#/components/features/conversation-panel/conversation-layouts-menu";
import { useConversationPanelPreferencesStore } from "#/stores/conversation-panel-preferences-store";
import { UNNAMED_AUTOMATION_FACET } from "#/components/features/conversation-panel/conversation-panel-list-helpers";

beforeEach(() => {
  useConversationPanelPreferencesStore.setState({
    organizeMode: "chronological",
    conversationSort: "updated",
    threadScope: "all",
    showOlderConversations: true,
    showArchivedConversations: false,
    showRepoBranchMetadata: false,
    showLlmProfiles: false,
    showTagsMetadata: false,
    showHoverMetadata: true,
    automationFilterMode: "all",
    selectedAutomationNames: [],
    selectedTagFacets: [],
  });
});

const renderMenu = (
  tagFacets: readonly string[] = [],
  automationNameFacets: readonly string[] = [],
) =>
  renderWithProviders(
    <ConversationLayoutsMenu
      menuOpen
      setMenuOpen={() => {}}
      menuRef={{ current: null }}
      backendKind="local"
      tagFacets={tagFacets}
      automationNameFacets={automationNameFacets}
      totalConversationsCount={1}
      onRequestDeleteAll={() => {}}
    />,
  );

describe("ConversationLayoutsMenu", () => {
  it("applies a layout preset bundle and marks it selected", async () => {
    const user = userEvent.setup();
    renderMenu();

    await user.click(screen.getByTestId("layout-preset-focused"));

    const state = useConversationPanelPreferencesStore.getState();
    expect(state.organizeMode).toBe("chronological");
    expect(state.conversationSort).toBe("updated");
    expect(state.threadScope).toBe("relevant");
    expect(state.showOlderConversations).toBe(false);
    // Applying a preset resets every layout-controlled field.
    expect(state.showTagsMetadata).toBe(true);
    expect(state.showHoverMetadata).toBe(true);
  });

  it("replaces the previous mode and marks each clicked preset selected", async () => {
    const user = userEvent.setup();
    renderMenu();

    await user.click(screen.getByTestId("layout-preset-minimal"));
    expect(screen.getByTestId("layout-preset-minimal")).toHaveAttribute(
      "aria-checked",
      "true",
    );

    await user.click(screen.getByTestId("layout-preset-focused"));
    expect(screen.getByTestId("layout-preset-focused")).toHaveAttribute(
      "aria-checked",
      "true",
    );
    expect(screen.getByTestId("layout-preset-minimal")).toHaveAttribute(
      "aria-checked",
      "false",
    );
    expect(
      useConversationPanelPreferencesStore.getState().showTagsMetadata,
    ).toBe(true);

    await user.click(screen.getByTestId("layout-preset-recent-activity"));
    expect(
      screen.getByTestId("layout-preset-recent-activity"),
    ).toHaveAttribute("aria-checked", "true");

    await user.click(screen.getByTestId("layout-preset-by-workspace"));
    expect(screen.getByTestId("layout-preset-by-workspace")).toHaveAttribute(
      "aria-checked",
      "true",
    );
  });

  it("labels the Advanced options row Custom when no preset matches", async () => {
    // Matches no preset: the chronological presets all hide older
    // conversations, and by-workspace requires grouped mode.
    useConversationPanelPreferencesStore.getState().applyLayoutSettings({
      organizeMode: "chronological",
      conversationSort: "updated",
      threadScope: "relevant",
      showOlderConversations: true,
    });

    renderMenu();

    expect(screen.getByTestId("advanced-options-row")).toHaveTextContent(
      "CONVERSATION_PANEL$ADVANCED_OPTIONS_CUSTOM",
    );
  });

  it("always exposes the Tag Filters section", () => {
    renderMenu(["project=vault"]);
    expect(screen.getByTestId("tag-filters-section")).toBeInTheDocument();
  });

  it("shows No visible tags when enabled with no facets, and toggles facets when present", async () => {
    const user = userEvent.setup();
    const { unmount } = renderMenu();
    await user.click(screen.getByTestId("tag-filters-section"));
    expect(screen.getByTestId("tag-filters-empty")).toBeInTheDocument();
    unmount();

    renderMenu(["project=vault"]);
    await user.click(screen.getByTestId("tag-filters-section"));
    await user.click(screen.getByTestId("tag-facet-row-project=vault"));
    expect(
      useConversationPanelPreferencesStore.getState().selectedTagFacets,
    ).toEqual(["project=vault"]);
  });

  it("opens the Advanced options modal, applies toggles live, and closes via Close", async () => {
    const user = userEvent.setup();
    renderMenu();

    expect(
      screen.queryByTestId("advanced-conversation-options-modal"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByTestId("advanced-options-row"));
    expect(
      await screen.findByTestId("advanced-conversation-options-modal"),
    ).toBeInTheDocument();

    // Toggle rows apply immediately and keep the modal open.
    await user.click(screen.getByTestId("toggle-tags-metadata"));
    expect(
      useConversationPanelPreferencesStore.getState().showTagsMetadata,
    ).toBe(true);
    expect(
      screen.getByTestId("advanced-conversation-options-modal"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("advanced-options-close"));
    expect(
      screen.queryByTestId("advanced-conversation-options-modal"),
    ).not.toBeInTheDocument();
  });

  it("exposes the automation-name facet rows once the scope is only-automations", async () => {
    // These rows are the only control for `selectedAutomationNames`, which is
    // persisted and narrows the list on its own — without them a stale
    // selection hides conversations with no way back.
    const user = userEvent.setup();
    renderMenu([], ["Nightly Audit", "PR Review Bot"]);

    await user.click(screen.getByTestId("advanced-options-row"));
    await screen.findByTestId("advanced-conversation-options-modal");

    // Hidden while the list is not scoped to automations...
    expect(
      screen.queryByTestId("automation-name-row-Nightly Audit"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByTestId("automation-filter-only"));

    // ...and selectable once it is.
    await user.click(
      await screen.findByTestId("automation-name-row-Nightly Audit"),
    );
    expect(
      useConversationPanelPreferencesStore.getState().selectedAutomationNames,
    ).toEqual(["Nightly Audit"]);

    // Toggling it back off leaves the scope alone.
    await user.click(screen.getByTestId("automation-name-row-Nightly Audit"));
    const next = useConversationPanelPreferencesStore.getState();
    expect(next.selectedAutomationNames).toEqual([]);
    expect(next.automationFilterMode).toBe("only-automations");
  });

  it("labels the unnamed-automation bucket instead of leaking its sentinel", async () => {
    const user = userEvent.setup();
    useConversationPanelPreferencesStore.setState({
      automationFilterMode: "only-automations",
    });
    renderMenu([], [UNNAMED_AUTOMATION_FACET]);

    await user.click(screen.getByTestId("advanced-options-row"));

    const row = await screen.findByTestId(
      `automation-name-row-${UNNAMED_AUTOMATION_FACET}`,
    );
    expect(row).not.toHaveTextContent(UNNAMED_AUTOMATION_FACET);
  });
});
