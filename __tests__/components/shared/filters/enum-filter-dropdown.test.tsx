import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { EnumFilterDropdown } from "#/components/shared/filters/enum-filter-dropdown";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { HideOlderConversationsRow } from "#/components/features/conversation-panel/hide-older-conversations-row";

const OPTIONS = ["status", "tag"] as const;

function Dropdown() {
  return (
    <EnumFilterDropdown
      testId="qa-group-by"
      value="status"
      onChange={vi.fn()}
      options={OPTIONS}
      labelByValue={{ status: "Group by status", tag: "Group by tag" }}
      ariaLabel="Group by"
    />
  );
}

describe("EnumFilterDropdown", () => {
  it("closes the open menu on Escape and returns focus to the trigger", async () => {
    // Arrange
    const user = userEvent.setup();
    render(<Dropdown />);
    const trigger = screen.getByRole("button", { name: "Group by" });
    await user.click(trigger);
    expect(screen.getByTestId("qa-group-by-menu")).toBeInTheDocument();

    // Act
    await user.keyboard("{Escape}");

    // Assert
    expect(screen.queryByTestId("qa-group-by-menu")).not.toBeInTheDocument();
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(trigger).toHaveFocus();
  });

  it("keeps a surrounding dialog open when Escape closes only the menu", async () => {
    // Arrange
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <ModalBackdrop onClose={onClose}>
        <Dropdown />
      </ModalBackdrop>,
    );
    await user.click(screen.getByRole("button", { name: "Group by" }));

    // Act: the first Escape closes the menu, the second the dialog
    await user.keyboard("{Escape}");

    // Assert
    expect(screen.queryByTestId("qa-group-by-menu")).not.toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();

    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe("HideOlderConversationsRow cutoff inside a dialog", () => {
  it("closes the cutoff menu first and the dialog on the next Escape", async () => {
    // Arrange
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <ModalBackdrop onClose={onClose}>
        <HideOlderConversationsRow />
      </ModalBackdrop>,
    );
    await user.click(
      within(screen.getByTestId("older-conversation-cutoff")).getByTestId(
        "dropdown-trigger",
      ),
    );
    expect(
      screen.getByTestId("older-conversation-cutoff-menu"),
    ).toBeInTheDocument();

    // Act + Assert
    await user.keyboard("{Escape}");
    expect(
      screen.queryByTestId("older-conversation-cutoff-menu"),
    ).not.toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();

    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
