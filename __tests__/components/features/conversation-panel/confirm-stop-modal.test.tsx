import React from "react";
import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { ConfirmStopModal } from "#/components/features/conversation-panel/confirm-stop-modal";

describe("ConfirmStopModal", () => {
  it("renders buttons with cancel-button and confirm-button test ids", () => {
    // Arrange: render the modal.
    renderWithProviders(
      <ConfirmStopModal onConfirm={vi.fn()} onCancel={vi.fn()} />,
    );

    // Assert: buttons expose the expected data-testid attributes.
    expect(screen.getByTestId("cancel-button")).toBeInTheDocument();
    expect(screen.getByTestId("confirm-button")).toBeInTheDocument();
  });
});
