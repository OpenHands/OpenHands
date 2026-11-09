import React from "react";
import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { ConfirmStopModal } from "#/components/features/conversation-panel/confirm-stop-modal";

const backendState = vi.hoisted(() => ({ kind: "local" as "local" | "cloud" }));

vi.mock("#/contexts/active-backend-context", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("#/contexts/active-backend-context")
  >()),
  useActiveBackend: () => ({ backend: { id: "b", kind: backendState.kind } }),
}));

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

  it.each([
    [
      "cloud",
      "CONVERSATION$CONFIRM_STOP_SANDBOX",
      "CONVERSATION$STOP_SANDBOX_WARNING",
      "COMMON$STOP_SANDBOX",
    ],
    [
      "local",
      "CONVERSATION$CONFIRM_CLOSE_CONVERSATION",
      "CONVERSATION$CLOSE_CONVERSATION_WARNING",
      "ACTION$CONFIRM_CLOSE",
    ],
  ] as const)(
    "uses the %s backend's wording",
    (kind, title, description, confirm) => {
      backendState.kind = kind;
      renderWithProviders(
        <ConfirmStopModal onConfirm={vi.fn()} onCancel={vi.fn()} />,
      );

      expect(screen.getByText(title)).toBeInTheDocument();
      expect(screen.getByText(description)).toBeInTheDocument();
      expect(screen.getByTestId("confirm-button")).toHaveTextContent(confirm);
    },
  );
});
