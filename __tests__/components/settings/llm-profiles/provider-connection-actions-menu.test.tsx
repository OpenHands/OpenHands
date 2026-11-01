import { useRef, useState } from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { ProviderConnectionActionsMenu } from "#/components/features/settings/llm-profiles/provider-connection-actions-menu";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const translations: Record<string, string> = {
        SETTINGS$PROVIDER_CONNECTION_BULK_ADD: "Bulk add",
        BUTTON$EDIT: "Edit",
        BUTTON$DELETE: "Delete",
      };
      return translations[key] || key;
    },
  }),
}));

const defaultProps = {
  onAddModels: vi.fn(),
  onEdit: vi.fn(),
  onDelete: vi.fn(),
  onClose: vi.fn(),
};

describe("ProviderConnectionActionsMenu", () => {
  it("renders Bulk add, Edit, and Delete buttons", () => {
    render(<ProviderConnectionActionsMenu {...defaultProps} />);

    expect(
      screen.getByTestId("provider-connection-add-models"),
    ).toHaveTextContent("Bulk add");
    expect(screen.getByTestId("provider-connection-edit")).toHaveTextContent(
      "Edit",
    );
    expect(screen.getByTestId("provider-connection-delete")).toHaveTextContent(
      "Delete",
    );
  });

  it("has correct accessibility attributes", () => {
    render(<ProviderConnectionActionsMenu {...defaultProps} />);

    const menu = screen.getByRole("menu");
    expect(menu).toHaveAttribute("aria-orientation", "vertical");

    const menuItems = screen.getAllByRole("menuitem");
    expect(menuItems).toHaveLength(3);
  });

  it("calls onAddModels and onClose when Bulk add is clicked", async () => {
    const user = userEvent.setup();
    const handleAddModels = vi.fn();
    const handleClose = vi.fn();

    render(
      <ProviderConnectionActionsMenu
        {...defaultProps}
        onAddModels={handleAddModels}
        onClose={handleClose}
      />,
    );

    await user.click(screen.getByTestId("provider-connection-add-models"));

    expect(handleAddModels).toHaveBeenCalledTimes(1);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it("calls onEdit and onClose when Edit is clicked", async () => {
    const user = userEvent.setup();
    const handleEdit = vi.fn();
    const handleClose = vi.fn();

    render(
      <ProviderConnectionActionsMenu
        {...defaultProps}
        onEdit={handleEdit}
        onClose={handleClose}
      />,
    );

    await user.click(screen.getByTestId("provider-connection-edit"));

    expect(handleEdit).toHaveBeenCalledTimes(1);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it("calls onDelete and onClose when Delete is clicked", async () => {
    const user = userEvent.setup();
    const handleDelete = vi.fn();
    const handleClose = vi.fn();

    render(
      <ProviderConnectionActionsMenu
        {...defaultProps}
        onDelete={handleDelete}
        onClose={handleClose}
      />,
    );

    await user.click(screen.getByTestId("provider-connection-delete"));

    expect(handleDelete).toHaveBeenCalledTimes(1);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it("calls onClose when clicking outside the menu", () => {
    const handleClose = vi.fn();

    render(
      <div>
        <div data-testid="outside" />
        <ProviderConnectionActionsMenu
          {...defaultProps}
          onClose={handleClose}
        />
      </div>,
    );

    fireEvent.mouseDown(screen.getByTestId("outside"));
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it("does not call onClose when clicking on anchorRef element", () => {
    const handleClose = vi.fn();
    const anchorButton = document.createElement("button");
    document.body.appendChild(anchorButton);
    const anchorRef = { current: anchorButton };

    render(
      <ProviderConnectionActionsMenu
        {...defaultProps}
        onClose={handleClose}
        anchorRef={anchorRef}
      />,
    );

    fireEvent.mouseDown(anchorButton);
    expect(handleClose).not.toHaveBeenCalled();
    anchorButton.remove();
  });

  it("calls onClose when Escape key is pressed", () => {
    const handleClose = vi.fn();

    render(
      <ProviderConnectionActionsMenu {...defaultProps} onClose={handleClose} />,
    );

    fireEvent.keyDown(document, { key: "Escape" });
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  describe("keyboard navigation", () => {
    it("focuses the first item (Bulk add) on mount", () => {
      render(<ProviderConnectionActionsMenu {...defaultProps} />);
      expect(
        screen.getByTestId("provider-connection-add-models"),
      ).toHaveFocus();
    });

    it("navigates down with ArrowDown and wraps around", async () => {
      const user = userEvent.setup();
      render(<ProviderConnectionActionsMenu {...defaultProps} />);

      const bulkAdd = screen.getByTestId("provider-connection-add-models");
      const edit = screen.getByTestId("provider-connection-edit");
      const del = screen.getByTestId("provider-connection-delete");

      expect(bulkAdd).toHaveFocus();

      await user.keyboard("{ArrowDown}");
      expect(edit).toHaveFocus();

      await user.keyboard("{ArrowDown}");
      expect(del).toHaveFocus();

      await user.keyboard("{ArrowDown}");
      expect(bulkAdd).toHaveFocus();
    });

    it("navigates up with ArrowUp and wraps around", async () => {
      const user = userEvent.setup();
      render(<ProviderConnectionActionsMenu {...defaultProps} />);

      const bulkAdd = screen.getByTestId("provider-connection-add-models");
      const del = screen.getByTestId("provider-connection-delete");

      expect(bulkAdd).toHaveFocus();

      await user.keyboard("{ArrowUp}");
      expect(del).toHaveFocus();
    });

    it("closes menu when Tab is pressed", async () => {
      const handleClose = vi.fn();
      const user = userEvent.setup();
      render(
        <ProviderConnectionActionsMenu
          {...defaultProps}
          onClose={handleClose}
        />,
      );

      await user.keyboard("{Tab}");
      expect(handleClose).toHaveBeenCalledTimes(1);
    });
  });

  describe("keyboard navigation when anchored to a row trigger", () => {
    function AnchoredMenu() {
      const triggerRef = useRef<HTMLButtonElement>(null);
      const [open, setOpen] = useState(false);
      return (
        <>
          <button
            ref={triggerRef}
            type="button"
            data-testid="provider-connection-menu-trigger"
            onClick={() => setOpen((value) => !value)}
          />
          {open && (
            <ProviderConnectionActionsMenu
              {...defaultProps}
              anchorRef={triggerRef}
              onClose={() => setOpen(false)}
            />
          )}
          <button type="button" data-testid="next-control" />
        </>
      );
    }

    it("moves focus to Bulk add when opened with click", async () => {
      const user = userEvent.setup();
      render(<AnchoredMenu />);

      await user.click(screen.getByTestId("provider-connection-menu-trigger"));

      expect(
        screen.getByTestId("provider-connection-add-models"),
      ).toHaveFocus();
    });

    it("moves focus to Bulk add when opened with Enter key", async () => {
      const user = userEvent.setup();
      render(<AnchoredMenu />);

      screen.getByTestId("provider-connection-menu-trigger").focus();
      await user.keyboard("{Enter}");

      expect(
        screen.getByTestId("provider-connection-add-models"),
      ).toHaveFocus();
    });

    it("closes on Tab and returns focus to the trigger", async () => {
      const user = userEvent.setup();
      render(<AnchoredMenu />);

      await user.click(screen.getByTestId("provider-connection-menu-trigger"));
      expect(
        screen.getByTestId("provider-connection-add-models"),
      ).toHaveFocus();

      await user.keyboard("{Tab}");

      expect(
        screen.queryByTestId("provider-connection-actions-menu"),
      ).not.toBeInTheDocument();
      expect(
        screen.getByTestId("provider-connection-menu-trigger"),
      ).toHaveFocus();
    });

    it("closes on Escape and returns focus to the trigger", async () => {
      const user = userEvent.setup();
      render(<AnchoredMenu />);

      await user.click(screen.getByTestId("provider-connection-menu-trigger"));
      expect(
        screen.getByTestId("provider-connection-add-models"),
      ).toHaveFocus();

      await user.keyboard("{Escape}");

      expect(
        screen.queryByTestId("provider-connection-actions-menu"),
      ).not.toBeInTheDocument();
      expect(
        screen.getByTestId("provider-connection-menu-trigger"),
      ).toHaveFocus();
    });
  });
});
