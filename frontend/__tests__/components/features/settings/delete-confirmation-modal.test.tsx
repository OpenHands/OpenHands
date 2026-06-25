import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { DeleteConfirmationModal } from "#/components/features/settings/delete-confirmation-modal";

const onClose = vi.fn();
const onDelete = vi.fn();

beforeEach(() => {
  onClose.mockReset();
  onDelete.mockReset();
});

describe("DeleteConfirmationModal", () => {
  it("renders nothing when not open", () => {
    const { container } = render(
      <DeleteConfirmationModal
        isOpen={false}
        itemName="test-item"
        onClose={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("renders title and message with item name", () => {
    render(
      <DeleteConfirmationModal
        isOpen={true}
        itemName="my-marketplace"
        onClose={onClose}
        onDelete={onDelete}
      />,
    );

    expect(
      screen.getByText("SETTINGS$DELETE_CONFIRMATION_TITLE"),
    ).toBeInTheDocument();
    // The message contains the itemName via interpolation
    expect(
      screen.getByText(/SETTINGS\$DELETE_CONFIRMATION_MESSAGE/i),
    ).toBeInTheDocument();
  });

  it("calls onDelete when delete button is clicked", async () => {
    render(
      <DeleteConfirmationModal
        isOpen={true}
        itemName="test-item"
        onClose={onClose}
        onDelete={onDelete}
      />,
    );

    await userEvent.click(screen.getByTestId("confirm-delete-button"));

    expect(onDelete).toHaveBeenCalledTimes(1);
    expect(onClose).not.toHaveBeenCalled();
  });

  it("calls onClose when cancel button is clicked", async () => {
    render(
      <DeleteConfirmationModal
        isOpen={true}
        itemName="test-item"
        onClose={onClose}
        onDelete={onDelete}
      />,
    );

    await userEvent.click(screen.getByRole("button", { name: "BUTTON$CANCEL" }));

    expect(onClose).toHaveBeenCalledTimes(1);
    expect(onDelete).not.toHaveBeenCalled();
  });

  it("disables buttons when isDeleting is true", () => {
    render(
      <DeleteConfirmationModal
        isOpen={true}
        itemName="test-item"
        onClose={onClose}
        onDelete={onDelete}
        isDeleting={true}
      />,
    );

    expect(screen.getByTestId("confirm-delete-button")).toBeDisabled();
    expect(screen.getByRole("button", { name: "BUTTON$CANCEL" })).toBeDisabled();
  });
});
