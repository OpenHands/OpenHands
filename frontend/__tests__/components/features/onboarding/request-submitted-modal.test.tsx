import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { RequestSubmittedModal } from "#/components/features/onboarding/request-submitted-modal";

describe("RequestSubmittedModal", () => {
  const defaultProps = {
    onClose: vi.fn(),
  };

  it("should render the modal", () => {
    render(<RequestSubmittedModal {...defaultProps} />);

    expect(screen.getByTestId("request-submitted-modal")).toBeInTheDocument();
  });

  it("should render the title", () => {
    render(<RequestSubmittedModal {...defaultProps} />);

    expect(
      screen.getByText("ENTERPRISE$REQUEST_SUBMITTED_TITLE"),
    ).toBeInTheDocument();
  });

  it("should render the description", () => {
    render(<RequestSubmittedModal {...defaultProps} />);

    expect(
      screen.getByText("ENTERPRISE$REQUEST_SUBMITTED_DESCRIPTION"),
    ).toBeInTheDocument();
  });

  it("should render the Done button", () => {
    render(<RequestSubmittedModal {...defaultProps} />);

    expect(
      screen.getByRole("button", { name: "ENTERPRISE$DONE_BUTTON" }),
    ).toBeInTheDocument();
  });

  it("should render the close button", () => {
    render(<RequestSubmittedModal {...defaultProps} />);

    expect(
      screen.getByRole("button", { name: "MODAL$CLOSE_BUTTON_LABEL" }),
    ).toBeInTheDocument();
  });

  it("should call onClose when Done button is clicked", async () => {
    const mockOnClose = vi.fn();
    const user = userEvent.setup();

    render(<RequestSubmittedModal onClose={mockOnClose} />);

    const doneButton = screen.getByRole("button", {
      name: "ENTERPRISE$DONE_BUTTON",
    });
    await user.click(doneButton);

    expect(mockOnClose).toHaveBeenCalledTimes(1);
  });

  it("should call onClose when close button is clicked", async () => {
    const mockOnClose = vi.fn();
    const user = userEvent.setup();

    render(<RequestSubmittedModal onClose={mockOnClose} />);

    const closeButton = screen.getByRole("button", {
      name: "MODAL$CLOSE_BUTTON_LABEL",
    });
    await user.click(closeButton);

    expect(mockOnClose).toHaveBeenCalledTimes(1);
  });

  it("should call onClose when Escape key is pressed", async () => {
    const mockOnClose = vi.fn();
    const user = userEvent.setup();

    render(<RequestSubmittedModal onClose={mockOnClose} />);

    await user.keyboard("{Escape}");

    expect(mockOnClose).toHaveBeenCalledTimes(1);
  });

  it("should call onClose when backdrop is clicked", async () => {
    const mockOnClose = vi.fn();
    const user = userEvent.setup();

    render(<RequestSubmittedModal onClose={mockOnClose} />);

    // Click on the backdrop (the semi-transparent overlay)
    const backdrop = screen.getByRole("dialog").querySelector(".bg-black");
    if (backdrop) {
      await user.click(backdrop);
    }

    expect(mockOnClose).toHaveBeenCalledTimes(1);
  });

  it("should have proper accessibility attributes", () => {
    render(<RequestSubmittedModal {...defaultProps} />);

    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAttribute(
      "aria-label",
      "ENTERPRISE$REQUEST_SUBMITTED_TITLE",
    );
  });
});
