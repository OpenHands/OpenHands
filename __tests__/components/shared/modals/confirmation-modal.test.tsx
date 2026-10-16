/* eslint-disable i18next/no-literal-string -- test mocks intentionally use literal strings */
import React, { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";

describe("ConfirmationModal focus management", () => {
  it("moves initial focus into the confirmation modal upon opening", async () => {
    function TestComponent() {
      const [isOpen, setIsOpen] = useState(false);
      return (
        <div>
          <button
            type="button"
            data-testid="enable-switch"
            onClick={() => setIsOpen(true)}
          >
            Enable App
          </button>
          {isOpen && (
            <ConfirmationModal
              text="Do you trust this app?"
              onConfirm={vi.fn()}
              onCancel={() => setIsOpen(false)}
            />
          )}
        </div>
      );
    }

    const user = userEvent.setup();
    render(<TestComponent />);

    const enableSwitch = screen.getByTestId("enable-switch");
    enableSwitch.focus();
    expect(document.activeElement).toBe(enableSwitch);

    await user.keyboard(" ");

    const cancelButton = screen.getByTestId("cancel-button");
    await waitFor(() => {
      expect(document.activeElement).toBe(cancelButton);
    });
    expect(document.activeElement).not.toBe(enableSwitch);
  });

  it("traps focus within the modal during Tab and Shift+Tab navigation", async () => {
    function TestComponent() {
      const [isOpen, setIsOpen] = useState(true);
      return (
        <div>
          <button type="button" data-testid="background-update">
            Background Update
          </button>
          {isOpen && (
            <ConfirmationModal
              text="Confirm action?"
              onConfirm={vi.fn()}
              onCancel={() => setIsOpen(false)}
            />
          )}
        </div>
      );
    }

    const user = userEvent.setup();
    render(<TestComponent />);

    const modal = screen.getByTestId("confirmation-modal");
    expect(modal).toBeInTheDocument();
    const cancelButton = screen.getByTestId("cancel-button");
    const confirmButton = screen.getByTestId("confirm-button");
    const backgroundUpdate = screen.getByTestId("background-update");

    await waitFor(() => {
      expect(document.activeElement).toBe(cancelButton);
    });

    // Tab forward from Cancel -> Confirm
    await user.tab();
    expect(document.activeElement).toBe(confirmButton);
    expect(document.activeElement).not.toBe(backgroundUpdate);

    // Tab forward from Confirm -> Wrap to Cancel (never reaching backgroundUpdate)
    await user.tab();
    expect(document.activeElement).toBe(cancelButton);
    expect(document.activeElement).not.toBe(backgroundUpdate);

    // Shift+Tab backward from Cancel -> Wrap to Confirm
    await user.tab({ shift: true });
    expect(document.activeElement).toBe(confirmButton);
    expect(document.activeElement).not.toBe(backgroundUpdate);
  });

  it("keeps focus trapped inside modal when isConfirming is true (action buttons disabled)", async () => {
    function TestComponent() {
      return (
        <div>
          <button type="button" data-testid="background-update">
            Background Update
          </button>
          <ConfirmationModal
            text="Performing mutation..."
            onConfirm={vi.fn()}
            onCancel={vi.fn()}
            isConfirming={true}
          />
        </div>
      );
    }

    const user = userEvent.setup();
    render(<TestComponent />);

    const modal = screen.getByTestId("confirmation-modal");
    const backgroundUpdate = screen.getByTestId("background-update");

    // Tab should not escape to background button even when inside buttons are disabled
    await user.tab();
    expect(document.activeElement).not.toBe(backgroundUpdate);
    expect(
      document.activeElement === modal ||
        modal.contains(document.activeElement),
    ).toBe(true);

    await user.tab({ shift: true });
    expect(document.activeElement).not.toBe(backgroundUpdate);
  });

  it("restores focus to the opener element when canceled or dismissed", async () => {
    function TestComponent() {
      const [isOpen, setIsOpen] = useState(false);
      return (
        <div>
          <button
            type="button"
            data-testid="opener-button"
            onClick={() => setIsOpen(true)}
          >
            Open Modal
          </button>
          {isOpen && (
            <ConfirmationModal
              text="Are you sure?"
              onConfirm={vi.fn()}
              onCancel={() => setIsOpen(false)}
            />
          )}
        </div>
      );
    }

    const user = userEvent.setup();
    render(<TestComponent />);

    const openerButton = screen.getByTestId("opener-button");
    openerButton.focus();
    expect(document.activeElement).toBe(openerButton);

    await user.click(openerButton);
    const cancelButton = screen.getByTestId("cancel-button");
    await waitFor(() => {
      expect(document.activeElement).toBe(cancelButton);
    });

    // Cancel closes the modal and returns focus to openerButton
    await user.click(cancelButton);
    expect(screen.queryByTestId("confirmation-modal")).toBeNull();
    expect(document.activeElement).toBe(openerButton);
  });

  it("safely handles opener element being unmounted from DOM without throwing", async () => {
    function TestComponent() {
      const [showOpener, setShowOpener] = useState(true);
      const [isOpen, setIsOpen] = useState(false);
      return (
        <div>
          {showOpener && (
            <button
              type="button"
              data-testid="ephemeral-opener"
              onClick={() => {
                setIsOpen(true);
                setShowOpener(false); // remove from DOM
              }}
            >
              Ephemeral Opener
            </button>
          )}
          {isOpen && (
            <ConfirmationModal
              text="Are you sure?"
              onConfirm={vi.fn()}
              onCancel={() => setIsOpen(false)}
            />
          )}
        </div>
      );
    }

    const user = userEvent.setup();
    render(<TestComponent />);

    const opener = screen.getByTestId("ephemeral-opener");
    opener.focus();
    await user.click(opener);

    const cancelButton = screen.getByTestId("cancel-button");
    await waitFor(() => {
      expect(document.activeElement).toBe(cancelButton);
    });

    // Close modal when opener is gone
    expect(() => user.click(cancelButton)).not.toThrow();
  });

  it("allows normal page Tab navigation after modal closes with no leaked listener", async () => {
    function TestComponent() {
      const [isOpen, setIsOpen] = useState(true);
      return (
        <div>
          <button type="button" data-testid="first-page-btn">
            First
          </button>
          <button type="button" data-testid="second-page-btn">
            Second
          </button>
          {isOpen && (
            <ConfirmationModal
              text="Close me"
              onConfirm={vi.fn()}
              onCancel={() => setIsOpen(false)}
            />
          )}
        </div>
      );
    }

    const user = userEvent.setup();
    render(<TestComponent />);

    const cancelButton = screen.getByTestId("cancel-button");
    await waitFor(() => {
      expect(document.activeElement).toBe(cancelButton);
    });

    await user.click(cancelButton);
    expect(screen.queryByTestId("confirmation-modal")).toBeNull();

    // Now Tab moves freely between page buttons
    const firstBtn = screen.getByTestId("first-page-btn");
    const secondBtn = screen.getByTestId("second-page-btn");

    firstBtn.focus();
    expect(document.activeElement).toBe(firstBtn);
    await user.tab();
    expect(document.activeElement).toBe(secondBtn);
  });
});
