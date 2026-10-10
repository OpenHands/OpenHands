import { useRef, useState } from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { AgentProfileActionsMenu } from "#/components/features/settings/agent-profiles/agent-profile-actions-menu";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const translations: Record<string, string> = {
        SETTINGS$PROFILE_EDIT: "Edit",
        SETTINGS$PROFILE_SET_ACTIVE: "Set as active",
        BUTTON$DELETE: "Delete",
      };
      return translations[key] || key;
    },
  }),
}));

const defaultProps = {
  onEdit: vi.fn(),
  onSetActive: vi.fn(),
  onDelete: vi.fn(),
  isActive: false,
  isActivating: false,
  onClose: vi.fn(),
};

describe("AgentProfileActionsMenu", () => {
  it("renders Edit, Set Active, and Delete buttons", () => {
    render(<AgentProfileActionsMenu {...defaultProps} />);

    expect(screen.getByTestId("agent-profile-edit")).toHaveTextContent("Edit");
    expect(screen.getByTestId("agent-profile-set-active")).toHaveTextContent(
      "Set as active",
    );
    expect(screen.getByTestId("agent-profile-delete")).toHaveTextContent(
      "Delete",
    );
  });

  it("calls onEdit and onClose when Edit is clicked", async () => {
    const user = userEvent.setup();
    const handleEdit = vi.fn();
    const handleClose = vi.fn();

    render(
      <AgentProfileActionsMenu
        {...defaultProps}
        onEdit={handleEdit}
        onClose={handleClose}
      />,
    );

    await user.click(screen.getByTestId("agent-profile-edit"));

    expect(handleEdit).toHaveBeenCalledTimes(1);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it("calls onSetActive and onClose when Set Active is clicked", async () => {
    const user = userEvent.setup();
    const handleSetActive = vi.fn();
    const handleClose = vi.fn();

    render(
      <AgentProfileActionsMenu
        {...defaultProps}
        onSetActive={handleSetActive}
        onClose={handleClose}
      />,
    );

    await user.click(screen.getByTestId("agent-profile-set-active"));

    expect(handleSetActive).toHaveBeenCalledTimes(1);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it("disables Set Active when already active", () => {
    render(<AgentProfileActionsMenu {...defaultProps} isActive />);

    const setActiveButton = screen.getByTestId("agent-profile-set-active");
    expect(setActiveButton).toBeDisabled();
  });

  it("disables Set Active when activating", () => {
    render(<AgentProfileActionsMenu {...defaultProps} isActivating />);

    const setActiveButton = screen.getByTestId("agent-profile-set-active");
    expect(setActiveButton).toBeDisabled();
  });

  it("calls onDelete and onClose when Delete is clicked", async () => {
    const user = userEvent.setup();
    const handleDelete = vi.fn();
    const handleClose = vi.fn();

    render(
      <AgentProfileActionsMenu
        {...defaultProps}
        onDelete={handleDelete}
        onClose={handleClose}
      />,
    );

    await user.click(screen.getByTestId("agent-profile-delete"));

    expect(handleDelete).toHaveBeenCalledTimes(1);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  describe("keyboard navigation", () => {
    it("focuses first enabled item on mount", () => {
      render(<AgentProfileActionsMenu {...defaultProps} />);
      expect(screen.getByTestId("agent-profile-edit")).toHaveFocus();
    });

    it("navigates down and wraps with ArrowDown key on non-active row", async () => {
      const user = userEvent.setup();
      render(<AgentProfileActionsMenu {...defaultProps} />);

      await user.keyboard("{ArrowDown}");
      expect(screen.getByTestId("agent-profile-set-active")).toHaveFocus();

      await user.keyboard("{ArrowDown}");
      expect(screen.getByTestId("agent-profile-delete")).toHaveFocus();

      await user.keyboard("{ArrowDown}");
      expect(screen.getByTestId("agent-profile-edit")).toHaveFocus();
    });

    it("navigates up and wraps with ArrowUp key on non-active row", async () => {
      const user = userEvent.setup();
      render(<AgentProfileActionsMenu {...defaultProps} />);

      await user.keyboard("{ArrowUp}");
      expect(screen.getByTestId("agent-profile-delete")).toHaveFocus();
    });

    it("skips the disabled Set as active item in both directions on active row", async () => {
      const user = userEvent.setup();
      render(<AgentProfileActionsMenu {...defaultProps} isActive />);

      expect(screen.getByTestId("agent-profile-edit")).toHaveFocus();

      await user.keyboard("{ArrowDown}");
      expect(screen.getByTestId("agent-profile-delete")).toHaveFocus();

      await user.keyboard("{ArrowUp}");
      expect(screen.getByTestId("agent-profile-edit")).toHaveFocus();
    });
  });

  describe("keyboard navigation when anchored to a row trigger", () => {
    function AnchoredMenu({ isActive = false }: { isActive?: boolean }) {
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
              isActive={isActive}
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

    it("moves focus to Edit when opened with Enter, then arrows across the disabled item", async () => {
      const user = userEvent.setup();
      render(<AnchoredMenu isActive />);

      screen.getByTestId("agent-profile-menu-trigger").focus();
      await user.keyboard("{Enter}");
      expect(screen.getByTestId("agent-profile-edit")).toHaveFocus();

      await user.keyboard("{ArrowDown}");
      expect(screen.getByTestId("agent-profile-delete")).toHaveFocus();

      await user.keyboard("{ArrowUp}");
      expect(screen.getByTestId("agent-profile-edit")).toHaveFocus();
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
  });
});
