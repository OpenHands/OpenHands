import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { ConfirmStopModal } from "#/components/features/conversation-panel/confirm-stop-modal";

describe("ConfirmStopModal", () => {
  it("exposes stable test ids on its actions", () => {
    // Arrange
    renderWithProviders(
      <ConfirmStopModal onConfirm={vi.fn()} onCancel={vi.fn()} />,
    );

    // Act
    const cancel = screen.getByTestId("cancel-button");
    const confirm = screen.getByTestId("confirm-button");

    // Assert
    expect(cancel).toHaveTextContent("BUTTON$CANCEL");
    expect(confirm).toHaveTextContent("ACTION$CONFIRM_CLOSE");
  });
});
