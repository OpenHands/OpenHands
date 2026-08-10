import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { ProfileActionsMenu } from "#/components/features/settings/llm-profiles/profile-actions-menu";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const translations: Record<string, string> = {
        "SETTINGS$PROFILE_EDIT": "Edit",
        "BUTTON$RENAME": "Rename",
        "BUTTON$DUPLICATE": "Duplicate",
        "SETTINGS$PROFILE_SET_ACTIVE": "Set as active",
        "SETTINGS$PROFILE_SET_DEFAULT": "Set as default",
        "BUTTON$DELETE": "Delete",
      };
      return translations[key] || key;
    },
  }),
}));

const defaultProps = {
  onEdit: vi.fn(),
  onRename: vi.fn(),
  onDuplicate: vi.fn(),
  onSetActive: vi.fn(),
  onDelete: vi.fn(),
  isActive: false,
  isActivating: false,
  onClose: vi.fn(),
};

describe("ProfileActionsMenu", () => {
  it("renders Edit, Rename, Duplicate, Set Active, and Delete buttons", () => {
    render(<ProfileActionsMenu {...defaultProps} />);

    expect(screen.getByTestId("profile-edit")).toHaveTextContent("Edit");
    expect(screen.getByTestId("profile-rename")).toHaveTextContent("Rename");
    expect(screen.getByTestId("profile-duplicate")).toHaveTextContent(
      "Duplicate",
    );
    expect(screen.getByTestId("profile-set-active")).toHaveTextContent("Set as default");
    expect(screen.getByTestId("profile-delete")).toHaveTextContent("Delete");
  });

  it("renders Duplicate between Rename and Set as active", () => {
    render(<ProfileActionsMenu {...defaultProps} />);

    const items = screen.getAllByRole("menuitem");
    const testIds = items.map((item) => item.getAttribute("data-testid"));

    expect(testIds).toEqual([
      "profile-edit",
      "profile-rename",
      "profile-duplicate",
      "profile-set-active",
      "profile-delete",
    ]);
  });

  it("calls onDuplicate and onClose when Duplicate is clicked", async () => {
    const user = userEvent.setup();
    const handleDuplicate = vi.fn();
    const handleClose = vi.fn();

    render(
      <ProfileActionsMenu
        {...defaultProps}
        onDuplicate={handleDuplicate}
        onClose={handleClose}
      />,
    );

    await user.click(screen.getByTestId("profile-duplicate"));

    expect(handleDuplicate).toHaveBeenCalledTimes(1);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it("calls onEdit and onClose when Edit is clicked", async () => {
    const user = userEvent.setup();
    const handleEdit = vi.fn();
    const handleClose = vi.fn();

    render(
      <ProfileActionsMenu
        {...defaultProps}
        onEdit={handleEdit}
        onClose={handleClose}
      />,
    );

    await user.click(screen.getByTestId("profile-edit"));

    expect(handleEdit).toHaveBeenCalledTimes(1);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it("calls onRename and onClose when Rename is clicked", async () => {
    const user = userEvent.setup();
    const handleRename = vi.fn();
    const handleClose = vi.fn();

    render(
      <ProfileActionsMenu
        {...defaultProps}
        onRename={handleRename}
        onClose={handleClose}
      />,
    );

    await user.click(screen.getByTestId("profile-rename"));

    expect(handleRename).toHaveBeenCalledTimes(1);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it("calls onSetActive and onClose when Set Active is clicked", async () => {
    const user = userEvent.setup();
    const handleSetActive = vi.fn();
    const handleClose = vi.fn();

    render(
      <ProfileActionsMenu
        {...defaultProps}
        onSetActive={handleSetActive}
        onClose={handleClose}
      />,
    );

    await user.click(screen.getByTestId("profile-set-active"));

    expect(handleSetActive).toHaveBeenCalledTimes(1);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it("disables Set Active when already active", () => {
    render(<ProfileActionsMenu {...defaultProps} isActive />);

    const setActiveButton = screen.getByTestId("profile-set-active");
    expect(setActiveButton).toBeDisabled();
  });

  it("disables Set Active when activating", () => {
    render(<ProfileActionsMenu {...defaultProps} isActivating />);

    const setActiveButton = screen.getByTestId("profile-set-active");
    expect(setActiveButton).toBeDisabled();
  });

  it("keeps Delete enabled even when isActive is true", () => {
    // The active profile is deletable; useEnsureActiveProfile promotes another
    // profile to active afterwards so one is always active in local mode.
    render(<ProfileActionsMenu {...defaultProps} isActive />);

    expect(screen.getByTestId("profile-delete")).not.toBeDisabled();
  });

  it("enables Delete button when isActive is false", () => {
    render(<ProfileActionsMenu {...defaultProps} isActive={false} />);

    expect(screen.getByTestId("profile-delete")).not.toBeDisabled();
  });

  it("calls onDelete and onClose when Delete is clicked", async () => {
    const user = userEvent.setup();
    const handleDelete = vi.fn();
    const handleClose = vi.fn();

    render(
      <ProfileActionsMenu
        {...defaultProps}
        onDelete={handleDelete}
        onClose={handleClose}
      />,
    );

    await user.click(screen.getByTestId("profile-delete"));

    expect(handleDelete).toHaveBeenCalledTimes(1);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it("calls onClose when clicking outside the menu", () => {
    const handleClose = vi.fn();

    render(
      <div>
        <div data-testid="outside">Outside</div>
        <ProfileActionsMenu {...defaultProps} onClose={handleClose} />
      </div>,
    );

    fireEvent.mouseDown(screen.getByTestId("outside"));

    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it("calls onClose when Escape key is pressed", () => {
    const handleClose = vi.fn();

    render(<ProfileActionsMenu {...defaultProps} onClose={handleClose} />);

    fireEvent.keyDown(document, { key: "Escape" });

    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it("does not call onClose for other keys", () => {
    const handleClose = vi.fn();

    render(<ProfileActionsMenu {...defaultProps} onClose={handleClose} />);

    fireEvent.keyDown(document, { key: "Enter" });
    fireEvent.keyDown(document, { key: "ArrowDown" });

    expect(handleClose).not.toHaveBeenCalled();
  });

  it("has correct accessibility attributes", () => {
    render(<ProfileActionsMenu {...defaultProps} />);

    const menu = screen.getByRole("menu");
    expect(menu).toHaveAttribute("aria-orientation", "vertical");

    const menuItems = screen.getAllByRole("menuitem");
    expect(menuItems).toHaveLength(5);
  });

  it("styles Delete like other menu items", () => {
    render(<ProfileActionsMenu {...defaultProps} />);

    const deleteButton = screen.getByTestId("profile-delete");
    expect(deleteButton).not.toHaveAttribute("data-destructive");
    expect(deleteButton.className).not.toMatch(/text-red/);
  });

  it("does not call onClose when clicking inside the menu container", () => {
    const handleClose = vi.fn();

    render(<ProfileActionsMenu {...defaultProps} onClose={handleClose} />);

    const menu = screen.getByRole("menu");
    fireEvent.mouseDown(menu);

    expect(handleClose).not.toHaveBeenCalled();
  });

  describe("keyboard navigation", () => {
    it("focuses first item on mount", () => {
      render(<ProfileActionsMenu {...defaultProps} />);
      expect(screen.getByTestId("profile-edit")).toHaveFocus();
    });

    it("navigates down with ArrowDown key", async () => {
      const user = userEvent.setup();
      render(<ProfileActionsMenu {...defaultProps} />);

      await user.keyboard("{ArrowDown}");
      expect(screen.getByTestId("profile-rename")).toHaveFocus();
    });

    it("navigates up with ArrowUp key", async () => {
      const user = userEvent.setup();
      render(<ProfileActionsMenu {...defaultProps} />);

      // Start at first item (Edit), go down to Rename, then back up
      await user.keyboard("{ArrowDown}");
      await user.keyboard("{ArrowUp}");
      expect(screen.getByTestId("profile-edit")).toHaveFocus();
    });

    it("wraps focus from last to first item with ArrowDown", async () => {
      const user = userEvent.setup();
      render(<ProfileActionsMenu {...defaultProps} />);

      // Navigate to the last item (Delete)
      await user.keyboard("{ArrowDown}"); // Rename
      await user.keyboard("{ArrowDown}"); // Duplicate
      await user.keyboard("{ArrowDown}"); // Set Active
      await user.keyboard("{ArrowDown}"); // Delete
      expect(screen.getByTestId("profile-delete")).toHaveFocus();

      // Press down again to wrap to first
      await user.keyboard("{ArrowDown}");
      expect(screen.getByTestId("profile-edit")).toHaveFocus();
    });

    it("wraps focus from first to last item with ArrowUp", async () => {
      const user = userEvent.setup();
      render(<ProfileActionsMenu {...defaultProps} />);

      // First item is focused (Edit), press up to wrap to last
      await user.keyboard("{ArrowUp}");
      expect(screen.getByTestId("profile-delete")).toHaveFocus();
    });

    it("closes menu when Tab is pressed", async () => {
      const handleClose = vi.fn();
      const user = userEvent.setup();
      render(<ProfileActionsMenu {...defaultProps} onClose={handleClose} />);

      await user.keyboard("{Tab}");
      expect(handleClose).toHaveBeenCalled();
    });
  });
});
