import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { DeleteConfirmationModal } from "#/components/features/automations/delete-confirmation-modal";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, params?: Record<string, string>) => {
      const translations: Record<string, string> = {
        AUTOMATIONS$DELETE_CONFIRM_TITLE: "Delete automation",
        AUTOMATIONS$DELETE_CONFIRM_MESSAGE: params?.name
          ? `Delete "${params.name}"? This can't be undone.`
          : "Delete this automation?",
        AUTOMATIONS$CANCEL: "Cancel",
        AUTOMATIONS$DELETE: "Delete",
        BUTTON$CLOSE: "Close",
      };
      return translations[key] || key;
    },
  }),
}));

describe("DeleteConfirmationModal", () => {
  it("renders nothing when closed", () => {
    const { container } = render(
      <DeleteConfirmationModal
        automationName="Nightly build"
        isOpen={false}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("exposes dialog semantics matching its sibling turn-off modal", () => {
    // Regression: this modal previously had no role/aria-modal/aria-labelledby
    // at all, while TurnOffConfirmationModal (the less consequential of the
    // two actions) did — a screen reader user got worse signal on the more
    // irreversible action.
    render(
      <DeleteConfirmationModal
        automationName="Nightly build"
        isOpen
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAttribute(
      "aria-labelledby",
      "delete-automation-title",
    );
    expect(screen.getByText("Delete automation")).toHaveAttribute(
      "id",
      "delete-automation-title",
    );
  });

  it("shows the automation name in the confirmation message", () => {
    render(
      <DeleteConfirmationModal
        automationName="Nightly build"
        isOpen
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(
      screen.getByText('Delete "Nightly build"? This can\'t be undone.'),
    ).toBeInTheDocument();
  });

  it("calls onConfirm when Delete is clicked", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    render(
      <DeleteConfirmationModal
        automationName="Nightly build"
        isOpen
        onConfirm={onConfirm}
        onCancel={vi.fn()}
      />,
    );

    await user.click(screen.getByTestId("delete-automation-confirm"));
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it("calls onCancel when Cancel is clicked", async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    render(
      <DeleteConfirmationModal
        automationName="Nightly build"
        isOpen
        onConfirm={vi.fn()}
        onCancel={onCancel}
      />,
    );

    await user.click(screen.getByText("Cancel"));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("calls onCancel when the backdrop is clicked", async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    render(
      <DeleteConfirmationModal
        automationName="Nightly build"
        isOpen
        onConfirm={vi.fn()}
        onCancel={onCancel}
      />,
    );

    await user.click(screen.getByRole("presentation"));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });
});
