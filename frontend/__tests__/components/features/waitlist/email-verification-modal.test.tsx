import { render, screen } from "@testing-library/react";
import { it, describe, expect, vi, beforeEach } from "vitest";
import { EmailVerificationModal } from "#/components/features/waitlist/email-verification-modal";

describe("EmailVerificationModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the email verification message", () => {
    // Arrange & Act
    render(<EmailVerificationModal onClose={vi.fn()} />);

    // Assert
    expect(
      screen.getByText("AUTH$PLEASE_CHECK_EMAIL_TO_VERIFY"),
    ).toBeInTheDocument();
  });

  it("should render the TermsAndPrivacyNotice component", () => {
    // Arrange & Act
    render(<EmailVerificationModal onClose={vi.fn()} />);

    // Assert
    const termsSection = screen.getByTestId("terms-and-privacy-notice");
    expect(termsSection).toBeInTheDocument();
  });
});
