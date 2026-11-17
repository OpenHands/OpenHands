import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { EnumFilterDropdown } from "#/components/shared/filters/enum-filter-dropdown";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";

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
