import { createRef } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";

import {
  ConversationPanelFilterMenu,
  type ConversationPanelFilterMenuProps,
} from "#/components/features/conversation-panel/conversation-panel-filter-menu";

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
    showOlderConversations: false,
    toggleShowOlderConversations: vi.fn(),
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
});
