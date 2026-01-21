import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import TOSReview from "#/components/features/auth/tos-review";

describe("TOSReview", () => {
  const mockSetIsOrgSelected = vi.fn();
  const mockSetIsTOSReviewComplete = vi.fn();

  const renderComponent = () =>
    render(
      <TOSReview
        setIsOrgSelected={mockSetIsOrgSelected}
        setIsTOSReviewComplete={mockSetIsTOSReviewComplete}
      />,
    );

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render heading and instructions", () => {
    renderComponent();

    expect(screen.getByText("Review Terms Of Service")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Please review and accept our Terms of Service before continuing.",
      ),
    ).toBeInTheDocument();
  });

  it("should render TOS acceptance checkbox", () => {
    renderComponent();

    expect(
      screen.getByText("I accept the Terms of Service"),
    ).toBeInTheDocument();
    expect(screen.getByRole("checkbox")).toBeInTheDocument();
  });

  it("should render Next and Back buttons", () => {
    renderComponent();

    expect(screen.getByRole("button", { name: "Next" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Back" })).toBeInTheDocument();
  });

  it("should have disabled Next button when TOS is not accepted", () => {
    renderComponent();

    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).toBeDisabled();
  });

  it("should enable Next button when TOS is accepted", async () => {
    const user = userEvent.setup();
    renderComponent();

    const checkbox = screen.getByRole("checkbox");
    await user.click(checkbox);

    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).not.toBeDisabled();
  });

  it("should call setIsOrgSelected(false) when Back button is clicked", async () => {
    const user = userEvent.setup();
    renderComponent();

    const backButton = screen.getByRole("button", { name: "Back" });
    await user.click(backButton);

    expect(mockSetIsOrgSelected).toHaveBeenCalledWith(false);
  });

  it("should call setIsTOSReviewComplete(true) when Next is clicked with TOS accepted", async () => {
    const user = userEvent.setup();
    renderComponent();

    // Accept TOS first
    const checkbox = screen.getByRole("checkbox");
    await user.click(checkbox);

    // Click Next
    const nextButton = screen.getByRole("button", { name: "Next" });
    await user.click(nextButton);

    expect(mockSetIsTOSReviewComplete).toHaveBeenCalledWith(true);
  });

  it("should not call setIsTOSReviewComplete when Next is clicked without accepting TOS", async () => {
    const user = userEvent.setup();
    renderComponent();

    const nextButton = screen.getByRole("button", { name: "Next" });
    // Button is disabled, but try to click anyway
    await user.click(nextButton);

    expect(mockSetIsTOSReviewComplete).not.toHaveBeenCalled();
  });
});
