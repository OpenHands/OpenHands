import { beforeEach, describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { ConversationLayoutsMenu } from "#/components/features/conversation-panel/conversation-layouts-menu";
import { useConversationPanelPreferencesStore } from "#/stores/conversation-panel-preferences-store";

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
    tagFiltersEnabled: false,
  });
});

const renderMenu = (tagFacets: readonly string[] = []) =>
  renderWithProviders(
    <ConversationLayoutsMenu
      menuOpen
      setMenuOpen={() => {}}
      menuRef={{ current: null }}
      backendKind="local"
      tagFacets={tagFacets}
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
    // Fields the preset does not name stay at their defaults.
    expect(state.showHoverMetadata).toBe(true);
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

  it("hides the Tag Filters section until the preference is enabled", async () => {
    const user = userEvent.setup();
    renderMenu(["project=vault"]);

    expect(screen.queryByTestId("tag-filters-section")).not.toBeInTheDocument();

    // The gate lives in the Advanced options modal.
    await user.click(screen.getByTestId("advanced-options-row"));
    await user.click(screen.getByTestId("toggle-tag-filters"));

    expect(await screen.findByTestId("tag-filters-section")).toBeInTheDocument();
  });

  it("shows No visible tags when enabled with no facets, and toggles facets when present", async () => {
    const user = userEvent.setup();
    useConversationPanelPreferencesStore
      .getState()
      .setTagFiltersEnabled(true);

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
});
