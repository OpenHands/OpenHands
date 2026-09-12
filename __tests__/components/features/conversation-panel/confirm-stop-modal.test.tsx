import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { ConfirmStopModal } from "#/components/features/conversation-panel/confirm-stop-modal";

describe("ConfirmStopModal", () => {
  it("gives the dialog an accessible name matching its title", () => {
    renderWithProviders(
      <ConfirmStopModal onConfirm={vi.fn()} onCancel={vi.fn()} />,
    );

    expect(
      screen.getByRole("dialog", {
        name: "CONVERSATION$CONFIRM_CLOSE_CONVERSATION",
      }),
    ).toBeInTheDocument();
  });

  it("calls onConfirm when the confirm button is clicked", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    renderWithProviders(
      <ConfirmStopModal onConfirm={onConfirm} onCancel={vi.fn()} />,
    );

    await user.click(screen.getByText("ACTION$CONFIRM_CLOSE"));
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it("calls onCancel when the cancel button is clicked", async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    renderWithProviders(
      <ConfirmStopModal onConfirm={vi.fn()} onCancel={onCancel} />,
    );

    await user.click(screen.getByText("BUTTON$CANCEL"));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });
});
