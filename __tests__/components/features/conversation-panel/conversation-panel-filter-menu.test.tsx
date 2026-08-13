import { createRef } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";

import {
  ConversationPanelFilterMenu,
  type ConversationPanelFilterMenuProps,
} from "#/components/features/conversation-panel/conversation-panel-filter-menu";
import { UNNAMED_AUTOMATION_FACET } from "#/components/features/conversation-panel/conversation-panel-list-helpers";

// ConversationPanelFilterMenu composes the extracted MenuHeading / MenuRow /
// MenuSeparator components. Rendering it open exercises all three through
// their real consumer, so these tests double as the regression guard for the
// extraction.
function renderFilterMenu(
  overrides: Partial<ConversationPanelFilterMenuProps> = {},
) {
  const props: ConversationPanelFilterMenuProps = {
    filterMenuOpen: true,
    setFilterMenuOpen: vi.fn(),
    menuRef: createRef<HTMLDivElement>(),
    backendKind: "local",
    organizeMode: "grouped",
    setOrganizeMode: vi.fn(),
    conversationSort: "created",
    setConversationSort: vi.fn(),
    threadScope: "all",
    setThreadScope: vi.fn(),
    automationFilterMode: "all",
    setAutomationFilterMode: vi.fn(),
    selectedAutomationNames: [],
    onToggleAutomationName: vi.fn(),
    automationNameFacets: [],
    selectedTagFacets: [],
    onToggleTagFacet: vi.fn(),
    tagFacets: [],
    showOlderConversations: false,
    toggleShowOlderConversations: vi.fn(),
    showArchivedConversations: false,
    toggleShowArchivedConversations: vi.fn(),
    showRepoBranchMetadata: false,
    toggleShowRepoBranchMetadata: vi.fn(),
    showLlmProfiles: false,
    toggleShowLlmProfiles: vi.fn(),
    showTagsMetadata: false,
    toggleShowTagsMetadata: vi.fn(),
    showHoverMetadata: false,
    toggleShowHoverMetadata: vi.fn(),
    totalConversationsCount: 5,
    onRequestDeleteAll: vi.fn(),
    ...overrides,
  };
  render(<ConversationPanelFilterMenu {...props} />);
  return props;
}

describe("ConversationPanelFilterMenu", () => {
  it("renders the section headings and rows while open", () => {
    // Arrange + Act
    renderFilterMenu({ filterMenuOpen: true });

    // Assert: a MenuHeading and representative MenuRows are present.
    expect(
      screen.getByTestId("older-conversations-filter-menu"),
    ).toBeInTheDocument();
    expect(screen.getByText("CONVERSATION_PANEL$ORGANIZE")).toBeInTheDocument();
    expect(
      screen.getByTestId("toggle-older-conversations"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("delete-all-conversations")).toBeInTheDocument();
  });

  it("orders metadata toggles as repo/branch, model, then tags", () => {
    renderFilterMenu({ filterMenuOpen: true });

    const repo = screen.getByTestId("toggle-repo-branch-metadata");
    const model = screen.getByTestId("toggle-llm-profiles");
    const tags = screen.getByTestId("toggle-tags-metadata");

    expect(
      repo.compareDocumentPosition(model) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      model.compareDocumentPosition(tags) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("runs a row's action and closes the menu when the row is clicked", async () => {
    // Arrange
    const user = userEvent.setup();
    const props = renderFilterMenu({ filterMenuOpen: true });

    // Act
    await user.click(screen.getByTestId("toggle-llm-profiles"));

    // Assert
    expect(props.toggleShowLlmProfiles).toHaveBeenCalledTimes(1);
    expect(props.setFilterMenuOpen).toHaveBeenCalledWith(false);
  });

  it("disables the delete-all row when there are no conversations", () => {
    // Arrange + Act
    renderFilterMenu({ filterMenuOpen: true, totalConversationsCount: 0 });

    // Assert
    expect(screen.getByTestId("delete-all-conversations")).toBeDisabled();
  });

  it("selects an automation filter mode and closes the menu", async () => {
    // Arrange: default mode is "all", so the trigger shows no active dot.
    const user = userEvent.setup();
    const props = renderFilterMenu();
    expect(
      screen.queryByTestId("automation-filter-active-indicator"),
    ).not.toBeInTheDocument();

    // Act
    await user.click(screen.getByTestId("automation-filter-hide"));

    // Assert
    expect(props.setAutomationFilterMode).toHaveBeenCalledWith(
      "hide-automations",
    );
    expect(props.setFilterMenuOpen).toHaveBeenCalledWith(false);
  });

  it("hides the automation-name rows outside only-automations mode", () => {
    // Arrange + Act
    renderFilterMenu({
      automationFilterMode: "hide-automations",
      automationNameFacets: ["Nightly Audit"],
    });

    // Assert
    expect(
      screen.queryByTestId("automation-name-filter-Nightly Audit"),
    ).not.toBeInTheDocument();
  });

  it("toggles automation names without closing the menu in only-automations mode", async () => {
    // Arrange: only-automations mode also lights the trigger's active dot;
    // the unnamed bucket renders with its translated label.
    const user = userEvent.setup();
    const props = renderFilterMenu({
      automationFilterMode: "only-automations",
      automationNameFacets: ["Nightly Audit", UNNAMED_AUTOMATION_FACET],
      selectedAutomationNames: ["Nightly Audit"],
    });
    expect(
      screen.getByTestId("automation-filter-active-indicator"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("CONVERSATION_PANEL$AUTOMATION_UNNAMED"),
    ).toBeInTheDocument();

    // Act
    await user.click(
      screen.getByTestId(`automation-name-filter-${UNNAMED_AUTOMATION_FACET}`),
    );

    // Assert: the name toggles while the menu stays open for multi-select.
    expect(props.onToggleAutomationName).toHaveBeenCalledWith(
      UNNAMED_AUTOMATION_FACET,
    );
    expect(props.setFilterMenuOpen).not.toHaveBeenCalled();
  });

  it("toggles tag facets without closing the menu and lights the active indicator", async () => {
    // Arrange: a tag selection lights the trigger's active dot even when the
    // automation filter is in its default "all" mode.
    const user = userEvent.setup();
    const props = renderFilterMenu({
      tagFacets: ["origin=slack", "owner=alice"],
      selectedTagFacets: ["origin=slack"],
    });
    expect(
      screen.getByTestId("tag-filter-active-indicator"),
    ).toBeInTheDocument();

    // Act
    await user.click(screen.getByTestId("tag-filter-owner=alice"));

    // Assert: the facet toggles while the menu stays open for multi-select.
    expect(props.onToggleTagFacet).toHaveBeenCalledWith("owner=alice");
    expect(props.setFilterMenuOpen).not.toHaveBeenCalled();
  });

  it("omits the tags section when no tag facets are available", () => {
    // Arrange + Act
    renderFilterMenu({ tagFacets: [] });

    // Assert
    expect(screen.queryByTestId(/^tag-filter-/)).not.toBeInTheDocument();
  });
});
