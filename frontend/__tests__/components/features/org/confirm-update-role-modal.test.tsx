import React from "react";
import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { ConfirmUpdateRoleModal } from "#/components/features/org/confirm-update-role-modal";

vi.mock("react-i18next", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-i18next")>()),
  Trans: ({
    values,
    components,
  }: {
    values: { email: string; role: string };
    components: { email: React.ReactElement; role: React.ReactElement };
  }) => (
    <>
      {React.cloneElement(components.email, {}, values.email)}
      {React.cloneElement(components.role, {}, values.role)}
    </>
  ),
}));

describe("ConfirmUpdateRoleModal", () => {
  it("should display the member email and new role in the confirmation message", () => {
    // Arrange
    const memberEmail = "test@example.com";
    const newRole = "admin";

    // Act
    renderWithProviders(
      <ConfirmUpdateRoleModal
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
        memberEmail={memberEmail}
        newRole={newRole}
      />,
    );

    // Assert
    expect(screen.getByText(memberEmail)).toBeInTheDocument();
    expect(screen.getByText(newRole)).toBeInTheDocument();
  });

  it("should call onConfirm when the confirm button is clicked", async () => {
    // Arrange
    const user = userEvent.setup();
    const onConfirmMock = vi.fn();
    renderWithProviders(
      <ConfirmUpdateRoleModal
        onConfirm={onConfirmMock}
        onCancel={vi.fn()}
        memberEmail="test@example.com"
        newRole="admin"
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
      <ConfirmUpdateRoleModal
        onConfirm={vi.fn()}
        onCancel={onCancelMock}
        memberEmail="test@example.com"
        newRole="admin"
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
      <ConfirmUpdateRoleModal
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
        memberEmail="test@example.com"
        newRole="admin"
        isLoading
      />,
    );

    // Assert
    expect(screen.getByTestId("confirm-button")).toBeDisabled();
    expect(screen.getByTestId("cancel-button")).toBeDisabled();
    expect(screen.getByTestId("loading-spinner")).toBeInTheDocument();
  });
});
