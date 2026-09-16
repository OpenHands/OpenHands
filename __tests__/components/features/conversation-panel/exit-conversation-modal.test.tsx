import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { ExitConversationModal } from "#/components/features/conversation-panel/exit-conversation-modal";

describe("ExitConversationModal", () => {
  it("gives the dialog an accessible name matching its title", () => {
    renderWithProviders(
      <ExitConversationModal
        onConfirm={vi.fn()}
        onClose={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(
      screen.getByRole("dialog", { name: "CONVERSATION$EXIT_WARNING" }),
    ).toBeInTheDocument();
  });

  it("calls onConfirm when the confirm button is clicked", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    renderWithProviders(
      <ExitConversationModal
        onConfirm={onConfirm}
        onClose={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    await user.click(screen.getByText("ACTION$CONFIRM"));
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it("calls onClose when the cancel button is clicked", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    renderWithProviders(
      <ExitConversationModal
        onConfirm={vi.fn()}
        onClose={onClose}
        onCancel={vi.fn()}
      />,
    );

    await user.click(screen.getByText("BUTTON$CANCEL"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
