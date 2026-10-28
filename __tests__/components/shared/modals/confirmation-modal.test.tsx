import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";

function TestWrapper({
  isConfirming = false,
  onConfirm = vi.fn(),
  onCancel = vi.fn(),
}: {
  isConfirming?: boolean;
  onConfirm?: () => void;
  onCancel?: () => void;
}) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div>
      <button
        type="button"
        data-testid="opener-button"
        onClick={() => setIsOpen(true)}
      >
        Open modal
      </button>
      <button type="button" data-testid="outside-button">
        Background button
      </button>
      {isOpen && (
        <ConfirmationModal
          text="Are you sure you want to proceed?"
          onConfirm={onConfirm}
          onCancel={() => {
            onCancel();
            setIsOpen(false);
          }}
          isConfirming={isConfirming}
        />
      )}
    </div>
  );
}

describe("ConfirmationModal focus management", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("moves initial focus to the first focusable control inside the modal upon opening", async () => {
    const user = userEvent.setup();
    render(<TestWrapper />);

    const opener = screen.getByTestId("opener-button");
    opener.focus();
    expect(opener).toHaveFocus();

    await user.click(opener);

    const cancelButton = await screen.findByTestId("cancel-button");
    expect(cancelButton).toBeInTheDocument();
    expect(cancelButton).toHaveFocus();
    expect(opener).not.toHaveFocus();
  });

  it("traps focus inside the modal with Tab and Shift+Tab", async () => {
    const user = userEvent.setup();
    render(<TestWrapper />);

    const opener = screen.getByTestId("opener-button");
    await user.click(opener);

    const cancelButton = await screen.findByTestId("cancel-button");
    const confirmButton = screen.getByTestId("confirm-button");
    expect(cancelButton).toHaveFocus();

    // Tab moves focus from cancel to confirm
    await user.tab();
    expect(confirmButton).toHaveFocus();

    // Tab from confirm wraps around to cancel
    await user.tab();
    expect(cancelButton).toHaveFocus();

    // Shift+Tab from cancel wraps around to confirm
    await user.tab({ shift: true });
    expect(confirmButton).toHaveFocus();

    // Shift+Tab from confirm moves to cancel
    await user.tab({ shift: true });
    expect(cancelButton).toHaveFocus();

    // Background button is never focused
    expect(screen.getByTestId("outside-button")).not.toHaveFocus();
  });

  it("restores focus to the opener element when dismissed via Cancel", async () => {
    const user = userEvent.setup();
    render(<TestWrapper />);

    const opener = screen.getByTestId("opener-button");
    opener.focus();
    await user.click(opener);

    const cancelButton = await screen.findByTestId("cancel-button");
    expect(cancelButton).toHaveFocus();

    await user.click(cancelButton);
    expect(screen.queryByTestId("confirmation-modal")).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
  });

  it("restores focus to the opener element when dismissed via Escape", async () => {
    const user = userEvent.setup();
    render(<TestWrapper />);

    const opener = screen.getByTestId("opener-button");
    opener.focus();
    await user.click(opener);

    await screen.findByTestId("cancel-button");
    await user.keyboard("{Escape}");

    expect(screen.queryByTestId("confirmation-modal")).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
  });

  it("keeps focus trapped inside modal when isConfirming is true", async () => {
    const user = userEvent.setup();
    render(<TestWrapper isConfirming />);

    const opener = screen.getByTestId("opener-button");
    await user.click(opener);

    const modal = await screen.findByTestId("confirmation-modal");
    expect(modal).toBeInTheDocument();

    const cancelButton = screen.getByTestId("cancel-button");
    const confirmButton = screen.getByTestId("confirm-button");
    expect(cancelButton).toBeDisabled();
    expect(confirmButton).toBeDisabled();

    // Tab does not escape to outside-button
    await user.tab();
    expect(screen.getByTestId("outside-button")).not.toHaveFocus();

    await user.tab({ shift: true });
    expect(screen.getByTestId("outside-button")).not.toHaveFocus();
  });

  it("handles unmounting without error when opener is removed from DOM", async () => {
    const user = userEvent.setup();

    function DetachedOpenerWrapper() {
      const [isOpen, setIsOpen] = useState(false);
      const [showOpener, setShowOpener] = useState(true);

      return (
        <div>
          {showOpener && (
            <button
              type="button"
              data-testid="temporary-opener"
              onClick={() => {
                setIsOpen(true);
                setShowOpener(false);
              }}
            >
              Open and remove me
            </button>
          )}
          {isOpen && (
            <ConfirmationModal
              text="Confirm action"
              onConfirm={() => setIsOpen(false)}
              onCancel={() => setIsOpen(false)}
            />
          )}
        </div>
      );
    }

    render(<DetachedOpenerWrapper />);
    const opener = screen.getByTestId("temporary-opener");
    opener.focus();
    await user.click(opener);

    const cancelButton = await screen.findByTestId("cancel-button");
    expect(cancelButton).toBeInTheDocument();

    // Closing should not throw even though opener was detached
    await user.click(cancelButton);
    expect(screen.queryByTestId("confirmation-modal")).not.toBeInTheDocument();
  });

  it("cleans up keydown listener on unmount so page tab navigation continues normally", async () => {
    const user = userEvent.setup();
    render(<TestWrapper />);

    const opener = screen.getByTestId("opener-button");
    await user.click(opener);

    const cancelButton = await screen.findByTestId("cancel-button");
    await user.click(cancelButton);

    // After modal closes, tab moves naturally between page buttons
    const outside = screen.getByTestId("outside-button");
    opener.focus();
    expect(opener).toHaveFocus();
    await user.tab();
    expect(outside).toHaveFocus();
  });
});
