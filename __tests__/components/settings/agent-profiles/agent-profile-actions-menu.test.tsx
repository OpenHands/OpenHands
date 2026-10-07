import { useRef, useState } from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { AgentProfileActionsMenu } from "#/components/features/settings/agent-profiles/agent-profile-actions-menu";

const defaultProps = {
  onEdit: vi.fn(),
  onSetActive: vi.fn(),
  onDelete: vi.fn(),
  isActive: false,
  isActivating: false,
  onClose: vi.fn(),
};

describe("AgentProfileActionsMenu", () => {
  it("renders Edit, Set as active, and Delete", () => {
    render(<AgentProfileActionsMenu {...defaultProps} />);

    const items = screen.getAllByRole("menuitem");
    expect(items.map((item) => item.getAttribute("data-testid"))).toEqual([
      "agent-profile-edit",
      "agent-profile-set-active",
      "agent-profile-delete",
    ]);
  });

  it("disables Set as active when already active", () => {
    render(<AgentProfileActionsMenu {...defaultProps} isActive />);

    expect(screen.getByTestId("agent-profile-set-active")).toBeDisabled();
  });

  describe("keyboard navigation when anchored to a row trigger", () => {
    // Mirrors AgentProfileRow: the trigger toggles the menu, which is portaled
    // to <body> against the trigger and unmounted on close.
    function AnchoredMenu({
      isActive = false,
      isActivating = false,
      onEdit = defaultProps.onEdit,
    }: {
      isActive?: boolean;
      isActivating?: boolean;
      onEdit?: () => void;
    }) {
      const triggerRef = useRef<HTMLButtonElement>(null);
      const [open, setOpen] = useState(false);
      return (
        <>
          <button
            ref={triggerRef}
            type="button"
            data-testid="agent-profile-menu-trigger"
            onClick={() => setOpen((value) => !value)}
          />
          {open && (
            <AgentProfileActionsMenu
              {...defaultProps}
              onEdit={onEdit}
              isActive={isActive}
              isActivating={isActivating}
              anchorRef={triggerRef}
              onClose={() => setOpen(false)}
            />
          )}
          <button type="button" data-testid="next-page-control" />
        </>
      );
    }

    it("moves focus to Edit when opened with the mouse", async () => {
      const user = userEvent.setup();
      render(<AnchoredMenu />);

      await user.click(screen.getByTestId("agent-profile-menu-trigger"));

      expect(screen.getByTestId("agent-profile-edit")).toHaveFocus();
    });

    it("moves focus to Edit when opened with Enter on the trigger", async () => {
      const user = userEvent.setup();
      render(<AnchoredMenu />);

      screen.getByTestId("agent-profile-menu-trigger").focus();
      await user.keyboard("{Enter}");

      expect(screen.getByTestId("agent-profile-edit")).toHaveFocus();
    });

    it("arrows across the disabled Set as active item on the active row", async () => {
      const user = userEvent.setup();
      render(<AnchoredMenu isActive />);

      await user.click(screen.getByTestId("agent-profile-menu-trigger"));
      expect(screen.getByTestId("agent-profile-edit")).toHaveFocus();

      await user.keyboard("{ArrowDown}");
      expect(screen.getByTestId("agent-profile-delete")).toHaveFocus();

      await user.keyboard("{ArrowDown}");
      expect(screen.getByTestId("agent-profile-edit")).toHaveFocus();

      await user.keyboard("{ArrowUp}");
      expect(screen.getByTestId("agent-profile-delete")).toHaveFocus();
    });

    it("cycles through all three items with wrap on a non-active row", async () => {
      const user = userEvent.setup();
      render(<AnchoredMenu />);

      await user.click(screen.getByTestId("agent-profile-menu-trigger"));
      expect(screen.getByTestId("agent-profile-edit")).toHaveFocus();

      await user.keyboard("{ArrowDown}");
      expect(screen.getByTestId("agent-profile-set-active")).toHaveFocus();

      await user.keyboard("{ArrowDown}");
      expect(screen.getByTestId("agent-profile-delete")).toHaveFocus();

      await user.keyboard("{ArrowDown}");
      expect(screen.getByTestId("agent-profile-edit")).toHaveFocus();

      await user.keyboard("{ArrowUp}");
      expect(screen.getByTestId("agent-profile-delete")).toHaveFocus();
    });

    it("closes on Escape and returns focus to the trigger", async () => {
      const user = userEvent.setup();
      render(<AnchoredMenu />);
      await user.click(screen.getByTestId("agent-profile-menu-trigger"));

      await user.keyboard("{Escape}");

      expect(
        screen.queryByTestId("agent-profile-actions-menu"),
      ).not.toBeInTheDocument();
      expect(screen.getByTestId("agent-profile-menu-trigger")).toHaveFocus();
    });

    it("closes on Tab and moves focus to the control after the trigger", async () => {
      const user = userEvent.setup();
      render(<AnchoredMenu />);
      await user.click(screen.getByTestId("agent-profile-menu-trigger"));

      await user.keyboard("{Tab}");

      expect(
        screen.queryByTestId("agent-profile-actions-menu"),
      ).not.toBeInTheDocument();
      expect(screen.getByTestId("next-page-control")).toHaveFocus();
    });

    it("runs the focused action when Enter is pressed", async () => {
      const onEdit = vi.fn();
      const user = userEvent.setup();
      render(<AnchoredMenu onEdit={onEdit} />);
      await user.click(screen.getByTestId("agent-profile-menu-trigger"));

      await user.keyboard("{Enter}");

      expect(onEdit).toHaveBeenCalledTimes(1);
    });
  });
});
