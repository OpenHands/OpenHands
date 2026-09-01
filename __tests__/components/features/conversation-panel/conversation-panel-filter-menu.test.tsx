import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";

import {
  ConversationPanelFilterMenu,
  type ConversationPanelFilterMenuProps,
} from "#/components/features/conversation-panel/conversation-panel-filter-menu";
import { UNNAMED_AUTOMATION_FACET } from "#/components/features/conversation-panel/conversation-panel-list-helpers";

function createFilterMenuProps(
  overrides: Partial<ConversationPanelFilterMenuProps> = {},
): ConversationPanelFilterMenuProps {
  return {
    filterMenuOpen: true,
    setFilterMenuOpen: vi.fn(),
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
}

function renderFilterMenu(
  overrides: Partial<ConversationPanelFilterMenuProps> = {},
) {
  const props = createFilterMenuProps(overrides);
  render(<ConversationPanelFilterMenu {...props} />);
  return props;
}

async function openAdvancedOptions(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByTestId("conversation-layout-advanced"));
  expect(
    await screen.findByTestId("conversation-advanced-options"),
  ).toBeInTheDocument();
}

describe("ConversationPanelFilterMenu", () => {
  it("shows a compact layout menu without the long filter list", () => {
    renderFilterMenu({ filterMenuOpen: true });

    const menu = screen.getByTestId("older-conversations-filter-menu");
    expect(within(menu).getAllByRole("menuitemradio")).toHaveLength(3);
    expect(within(menu).getByRole("menuitem")).toBeInTheDocument();
    expect(
      screen.getByText("CONVERSATION_PANEL$BY_WORKSPACE"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("CONVERSATION_PANEL$CHRONOLOGICAL"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("conversation-layout-show-active"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("conversation-layout-advanced"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("delete-all-conversations"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("toggle-older-conversations"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("CONVERSATION_PANEL$ORGANIZE"),
    ).not.toBeInTheDocument();
  });

  it("applies an organize layout and closes the primary menu", async () => {
    const user = userEvent.setup();
    const props = renderFilterMenu({
      filterMenuOpen: true,
      organizeMode: "grouped",
    });

    await user.click(screen.getByText("CONVERSATION_PANEL$CHRONOLOGICAL"));

    expect(props.setOrganizeMode).toHaveBeenCalledWith("chronological");
    expect(props.setFilterMenuOpen).toHaveBeenCalledWith(false);
  });

  it("toggles show-active without changing organize mode", async () => {
    const user = userEvent.setup();
    const props = renderFilterMenu({
      filterMenuOpen: true,
      organizeMode: "grouped",
      threadScope: "all",
    });

    await user.click(screen.getByTestId("conversation-layout-show-active"));

    expect(props.setOrganizeMode).not.toHaveBeenCalled();
    expect(props.setThreadScope).toHaveBeenCalledWith("relevant");
    expect(props.setFilterMenuOpen).toHaveBeenCalledWith(false);
  });

  it("opens advanced options from the layout menu", async () => {
    const user = userEvent.setup();
    const props = renderFilterMenu({ filterMenuOpen: true });

    await openAdvancedOptions(user);

    expect(props.setFilterMenuOpen).not.toHaveBeenCalled();
    expect(
      screen.getByTestId("conversation-advanced-options"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("older-conversations-filter-menu"),
    ).not.toBeInTheDocument();
  });

  it("keeps advanced options open until a click outside the dropdown", async () => {
    const user = userEvent.setup();
    const setFilterMenuOpen = vi.fn();
    render(
      <div>
        <ConversationPanelFilterMenu
          {...createFilterMenuProps({ setFilterMenuOpen })}
        />
        <div data-testid="outside-filter-menu" />
      </div>,
    );

    await user.click(screen.getByTestId("conversation-layout-advanced"));

    expect(setFilterMenuOpen).not.toHaveBeenCalled();
    expect(
      screen.getByTestId("conversation-advanced-options"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("outside-filter-menu"));

    expect(setFilterMenuOpen).toHaveBeenCalledWith(false);
  });

  it("orders metadata toggles as repo/branch, model, then tags", async () => {
    const user = userEvent.setup();
    renderFilterMenu({ filterMenuOpen: true });
    await openAdvancedOptions(user);

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

  it("closes advanced options after applying a setting", async () => {
    const user = userEvent.setup();
    const props = renderFilterMenu({ filterMenuOpen: true });
    await openAdvancedOptions(user);

    await user.click(screen.getByTestId("toggle-llm-profiles"));

    expect(props.toggleShowLlmProfiles).toHaveBeenCalledTimes(1);
    expect(props.setFilterMenuOpen).toHaveBeenCalledWith(false);
  });

  it("disables the delete-all row when there are no conversations", async () => {
    const user = userEvent.setup();
    renderFilterMenu({ filterMenuOpen: true, totalConversationsCount: 0 });
    await openAdvancedOptions(user);

    expect(screen.getByTestId("delete-all-conversations")).toBeDisabled();
  });

  it("selects an automation filter mode from advanced options", async () => {
    const user = userEvent.setup();
    const props = renderFilterMenu();
    expect(
      screen.queryByTestId("automation-filter-active-indicator"),
    ).not.toBeInTheDocument();

    await openAdvancedOptions(user);
    await user.click(screen.getByTestId("automation-filter-hide"));

    expect(props.setAutomationFilterMode).toHaveBeenCalledWith(
      "hide-automations",
    );
    expect(props.setFilterMenuOpen).toHaveBeenCalledWith(false);
  });

  it("hides the automation-name rows outside only-automations mode", async () => {
    const user = userEvent.setup();
    renderFilterMenu({
      automationFilterMode: "hide-automations",
      automationNameFacets: ["Nightly Audit"],
    });
    await openAdvancedOptions(user);

    expect(
      screen.queryByTestId("automation-name-filter-Nightly Audit"),
    ).not.toBeInTheDocument();
  });

  it("toggles automation names in only-automations mode", async () => {
    const user = userEvent.setup();
    const props = renderFilterMenu({
      automationFilterMode: "only-automations",
      automationNameFacets: ["Nightly Audit", UNNAMED_AUTOMATION_FACET],
      selectedAutomationNames: ["Nightly Audit"],
    });
    expect(
      screen.getByTestId("automation-filter-active-indicator"),
    ).toBeInTheDocument();

    await openAdvancedOptions(user);
    expect(
      screen.getByText("CONVERSATION_PANEL$AUTOMATION_UNNAMED"),
    ).toBeInTheDocument();

    await user.click(
      screen.getByTestId(`automation-name-filter-${UNNAMED_AUTOMATION_FACET}`),
    );

    expect(props.onToggleAutomationName).toHaveBeenCalledWith(
      UNNAMED_AUTOMATION_FACET,
    );
    expect(props.setFilterMenuOpen).toHaveBeenCalledWith(false);
  });
});
