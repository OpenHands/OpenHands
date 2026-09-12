import React from "react";
import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { ConfirmDeleteModal } from "#/components/features/conversation-panel/confirm-delete-modal";

vi.mock("react-i18next", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-i18next")>()),
  Trans: ({
    values,
    components,
  }: {
    values: { title: string };
    components: { title: React.ReactElement };
  }) => React.cloneElement(components.title, {}, values.title),
}));

describe("ConfirmDeleteModal", () => {
  it("gives the dialog an accessible name matching its title, on the app's most destructive confirmation", () => {
    // Regression: ModalBackdrop supports aria-label for exactly this, but
    // this modal (and 4 siblings in this feature area) never passed it — a
    // screen reader announced a bare, unnamed "dialog" on delete/archive/
    // stop/exit confirmations, the flows where knowing what's being
    // confirmed matters most.
    renderWithProviders(
      <ConfirmDeleteModal onConfirm={vi.fn()} onCancel={vi.fn()} />,
    );

    expect(
      screen.getByRole("dialog", { name: "CONVERSATION$CONFIRM_DELETE" }),
    ).toBeInTheDocument();
  });

  it("should display the conversation title", () => {
    renderWithProviders(
      <ConfirmDeleteModal
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
        conversationTitle="My Test Conversation"
      />,
    );

    expect(screen.getByText(/My Test Conversation/)).toBeInTheDocument();
  });

  it("falls back to the default warning when description is null", () => {
    renderWithProviders(
      <ConfirmDeleteModal
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
        description={null}
      />,
    );

    expect(screen.getByText("CONVERSATION$DELETE_WARNING")).toBeInTheDocument();
  });

  it("places Cancel before Confirm in the footer so the dominant action is the last focusable button", () => {
    // Arrange: render the modal so both footer buttons are mounted.
    renderWithProviders(
      <ConfirmDeleteModal onConfirm={vi.fn()} onCancel={vi.fn()} />,
    );

    // Act: locate both footer buttons.
    const cancel = screen.getByText("BUTTON$CANCEL");
    const confirm = screen.getByText("ACTION$CONFIRM_DELETE");

    // Assert: Cancel precedes the dominant Confirm action in DOM order.

    expect(
      cancel.compareDocumentPosition(confirm) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });
});
