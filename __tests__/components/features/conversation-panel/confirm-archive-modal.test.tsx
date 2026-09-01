import React from "react";
import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { ConfirmArchiveModal } from "#/components/features/conversation-panel/confirm-archive-modal";

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

describe("ConfirmArchiveModal", () => {
  it("should display the conversation title in the warning", () => {
    renderWithProviders(
      <ConfirmArchiveModal
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
        conversationTitle="My Test Conversation"
      />,
    );

    expect(screen.getByText(/My Test Conversation/)).toBeInTheDocument();
  });

  it("falls back to the default warning when no title is provided", () => {
    renderWithProviders(
      <ConfirmArchiveModal onConfirm={vi.fn()} onCancel={vi.fn()} />,
    );

    expect(
      screen.getByText("CONVERSATION$ARCHIVE_WARNING"),
    ).toBeInTheDocument();
  });

  it("places Cancel before Confirm in the footer", () => {
    renderWithProviders(
      <ConfirmArchiveModal onConfirm={vi.fn()} onCancel={vi.fn()} />,
    );

    const cancel = screen.getByText("BUTTON$CANCEL");
    const confirm = screen.getByText("ACTION$CONFIRM_ARCHIVE");

    // eslint-disable-next-line no-bitwise
    expect(
      cancel.compareDocumentPosition(confirm) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });
});
