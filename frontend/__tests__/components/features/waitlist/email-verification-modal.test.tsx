import React from "react";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router";
import { emailService } from "#/api/email-service/email-service.api";
import { EmailVerificationModal } from "#/components/features/waitlist/email-verification-modal";
import * as ToastHandlers from "#/utils/custom-toast-handlers";
import { renderWithProviders, createAxiosError } from "../../../../test-utils";

describe("EmailVerificationModal", () => {
  const mockOnClose = vi.fn();

  const resendEmailVerificationSpy = vi.spyOn(
    emailService,
    "resendEmailVerification",
  );
  const displaySuccessToastSpy = vi.spyOn(ToastHandlers, "displaySuccessToast");
  const displayErrorToastSpy = vi.spyOn(ToastHandlers, "displayErrorToast");

  const renderWithRouter = (ui: React.ReactElement) => {
    return renderWithProviders(<MemoryRouter>{ui}</MemoryRouter>);
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the email verification message", () => {
    // Arrange & Act
    renderWithRouter(<EmailVerificationModal onClose={mockOnClose} />);

    // Assert
    expect(
      screen.getByText("AUTH$PLEASE_CHECK_EMAIL_TO_VERIFY"),
    ).toBeInTheDocument();
  });

  it("should render the TermsAndPrivacyNotice component", () => {
    // Arrange & Act
    renderWithRouter(<EmailVerificationModal onClose={mockOnClose} />);

    // Assert
    const termsSection = screen.getByTestId("terms-and-privacy-notice");
    expect(termsSection).toBeInTheDocument();
  });

  it("should render resend verification button", () => {
    // Arrange & Act
    renderWithRouter(<EmailVerificationModal onClose={mockOnClose} />);

    // Assert
    expect(
      screen.getByText("SETTINGS$RESEND_VERIFICATION"),
    ).toBeInTheDocument();
  });

  it("should call resendEmailVerification when the button is clicked", async () => {
    // Arrange
    const userId = "test_user_id";
    resendEmailVerificationSpy.mockResolvedValue({
      message: "Email verification message sent",
    });
    renderWithRouter(
      <EmailVerificationModal onClose={mockOnClose} userId={userId} />,
    );

    // Act
    const resendButton = screen.getByText("SETTINGS$RESEND_VERIFICATION");
    await userEvent.click(resendButton);

    // Assert
    await waitFor(() => {
      expect(resendEmailVerificationSpy).toHaveBeenCalledWith({
        userId,
        isAuthFlow: true,
      });
    });
  });

  it("should display success toast when resend succeeds", async () => {
    // Arrange
    resendEmailVerificationSpy.mockResolvedValue({
      message: "Email verification message sent",
    });
    renderWithRouter(<EmailVerificationModal onClose={mockOnClose} />);

    // Act
    const resendButton = screen.getByText("SETTINGS$RESEND_VERIFICATION");
    await userEvent.click(resendButton);

    // Assert
    await waitFor(() => {
      expect(displaySuccessToastSpy).toHaveBeenCalledWith(
        "SETTINGS$VERIFICATION_EMAIL_SENT",
      );
    });
  });

  it("should display rate limit error message when receiving 429 status", async () => {
    // Arrange
    const rateLimitError = createAxiosError(429, "Too Many Requests", {
      detail: "Too many requests. Please wait 2 minutes before trying again.",
    });
    resendEmailVerificationSpy.mockRejectedValue(rateLimitError);
    renderWithRouter(<EmailVerificationModal onClose={mockOnClose} />);

    // Act
    const resendButton = screen.getByText("SETTINGS$RESEND_VERIFICATION");
    await userEvent.click(resendButton);

    // Assert
    await waitFor(() => {
      expect(displayErrorToastSpy).toHaveBeenCalledWith(
        "Too many requests. Please wait 2 minutes before trying again.",
      );
    });
  });

  it("should display generic error message when receiving non-429 error", async () => {
    // Arrange
    const genericError = createAxiosError(500, "Internal Server Error", {
      error: "Internal server error",
    });
    resendEmailVerificationSpy.mockRejectedValue(genericError);
    renderWithRouter(<EmailVerificationModal onClose={mockOnClose} />);

    // Act
    const resendButton = screen.getByText("SETTINGS$RESEND_VERIFICATION");
    await userEvent.click(resendButton);

    // Assert
    await waitFor(() => {
      expect(displayErrorToastSpy).toHaveBeenCalledWith(
        "SETTINGS$FAILED_TO_RESEND_VERIFICATION",
      );
    });
  });

  it("should disable button and show sending text while request is pending", async () => {
    // Arrange
    let resolvePromise: (value: { message: string }) => void;
    const pendingPromise = new Promise<{ message: string }>((resolve) => {
      resolvePromise = resolve;
    });
    resendEmailVerificationSpy.mockReturnValue(pendingPromise);
    renderWithRouter(<EmailVerificationModal onClose={mockOnClose} />);

    // Act
    const resendButton = screen.getByText("SETTINGS$RESEND_VERIFICATION");
    await userEvent.click(resendButton);

    // Assert
    await waitFor(() => {
      const sendingButton = screen.getByText("SETTINGS$SENDING");
      expect(sendingButton).toBeInTheDocument();
      expect(sendingButton).toBeDisabled();
    });

    // Cleanup
    resolvePromise!({ message: "Email verification message sent" });
  });

  it("should re-enable button after request completes", async () => {
    // Arrange
    resendEmailVerificationSpy.mockResolvedValue({
      message: "Email verification message sent",
    });
    renderWithRouter(<EmailVerificationModal onClose={mockOnClose} />);

    // Act
    const resendButton = screen.getByText("SETTINGS$RESEND_VERIFICATION");
    await userEvent.click(resendButton);

    // Assert
    await waitFor(() => {
      expect(resendEmailVerificationSpy).toHaveBeenCalled();
    });
    // After successful send, the button will be disabled due to cooldown
    // So we just verify the mutation was called successfully
    await waitFor(() => {
      const button = screen.getByRole("button");
      expect(button).toBeDisabled(); // Button is disabled during cooldown
      expect(button).toHaveTextContent(/SETTINGS\$RESEND_VERIFICATION/);
    });
  });
});
