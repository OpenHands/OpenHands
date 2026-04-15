import React from "react";
import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { ConfirmRemoveMemberModal } from "#/components/features/org/confirm-remove-member-modal";

vi.mock("react-i18next", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-i18next")>()),
  Trans: ({
    values,
    components,
  }: {
    values: { email: string };
    components: { email: React.ReactElement };
  }) => React.cloneElement(components.email, {}, values.email),
}));

describe("ConfirmRemoveMemberModal", () => {
  it("should display the member email in the confirmation message", () => {
    // Arrange
    const memberEmail = "test@example.com";

    // Act
    renderWithProviders(
      <ConfirmRemoveMemberModal
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
        memberEmail={memberEmail}
      />,
    );

    // Assert
    expect(screen.getByText(memberEmail)).toBeInTheDocument();
  });

  it("should call onConfirm when the confirm button is clicked", async () => {
    // Arrange
    const user = userEvent.setup();
    const onConfirmMock = vi.fn();
    renderWithProviders(
      <ConfirmRemoveMemberModal
        onConfirm={onConfirmMock}
        onCancel={vi.fn()}
        memberEmail="test@example.com"
      />,
    );

    // Act
    await user.click(screen.getByTestId("confirm-button"));

    // Assert
    expect(onConfirmMock).toHaveBeenCalledOnce();
  });

  it("should call onCancel when the cancel button is clicked", async () => {
    // Arrange
    const user = userEvent.setup();
    const onCancelMock = vi.fn();
    renderWithProviders(
      <ConfirmRemoveMemberModal
        onConfirm={vi.fn()}
        onCancel={onCancelMock}
        memberEmail="test@example.com"
      />,
    );

    // Act
    await user.click(screen.getByTestId("cancel-button"));

    // Assert
    expect(onCancelMock).toHaveBeenCalledOnce();
  });

  it("should disable buttons and show loading spinner when isLoading is true", () => {
    // Arrange & Act
    renderWithProviders(
      <ConfirmRemoveMemberModal
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
        memberEmail="test@example.com"
        isLoading
      />,
    );

    // Assert
    expect(screen.getByTestId("confirm-button")).toBeDisabled();
    expect(screen.getByTestId("cancel-button")).toBeDisabled();
    expect(screen.getByTestId("loading-spinner")).toBeInTheDocument();
  });
});
