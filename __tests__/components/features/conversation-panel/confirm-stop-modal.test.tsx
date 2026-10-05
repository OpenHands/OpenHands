import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { ConfirmStopModal } from "#/components/features/conversation-panel/confirm-stop-modal";

describe("ConfirmStopModal", () => {
  it("exposes test ids on the cancel and confirm buttons", () => {
    renderWithProviders(
      <ConfirmStopModal onConfirm={vi.fn()} onCancel={vi.fn()} />,
    );

    expect(screen.getByTestId("cancel-button")).toBeInTheDocument();
    expect(screen.getByTestId("confirm-button")).toBeInTheDocument();
  });
});
