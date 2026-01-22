import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import EnterEmailOTP from "#/components/features/auth/otp-login/enter-email-otp";
import { I18nKey } from "#/i18n/declaration";

describe("EnterEmailOTP", () => {
  const mockSetIsLogginInWithEmail = vi.fn();
  const mockSetIsReadyToVerifyEmail = vi.fn();

  const renderComponent = () =>
    render(
      <EnterEmailOTP
        setIsLogginInWithEmail={mockSetIsLogginInWithEmail}
        setIsReadyToVerifyEmail={mockSetIsReadyToVerifyEmail}
      />,
    );

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render heading and email input", () => {
    renderComponent();

    expect(screen.getByText(I18nKey.AUTH$SIGN_IN_WITH_EMAIL)).toBeInTheDocument();
    expect(screen.getByText(I18nKey.AUTH$YOUR_EMAIL_ADDRESS)).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText("name@email.com"),
    ).toBeInTheDocument();
  });

  it("should render verify and cancel buttons", () => {
    renderComponent();

    expect(screen.getByRole("button", { name: I18nKey.AUTH$VERIFY })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: I18nKey.AUTH$CANCEL })).toBeInTheDocument();
  });

  it("should call setIsLogginInWithEmail(false) when cancel button is clicked", async () => {
    const user = userEvent.setup();
    renderComponent();

    const cancelButton = screen.getByRole("button", { name: I18nKey.AUTH$CANCEL });
    await user.click(cancelButton);

    expect(mockSetIsLogginInWithEmail).toHaveBeenCalledWith(false);
  });

  it("should show error when email is empty and verify is clicked", async () => {
    const user = userEvent.setup();
    renderComponent();

    const verifyButton = screen.getByRole("button", { name: I18nKey.AUTH$VERIFY });
    await user.click(verifyButton);

    expect(screen.getByText(I18nKey.AUTH$EMAIL_REQUIRED)).toBeInTheDocument();
    expect(mockSetIsReadyToVerifyEmail).not.toHaveBeenCalled();
  });

  it("should show error when email format is invalid", async () => {
    const user = userEvent.setup();
    renderComponent();

    const emailInput = screen.getByPlaceholderText("name@email.com");
    await user.type(emailInput, "invalid-email");

    const verifyButton = screen.getByRole("button", { name: I18nKey.AUTH$VERIFY });
    await user.click(verifyButton);

    expect(screen.getByText(I18nKey.AUTH$INVALID_EMAIL_FORMAT)).toBeInTheDocument();
    expect(mockSetIsReadyToVerifyEmail).not.toHaveBeenCalled();
  });

  it("should call setIsReadyToVerifyEmail(true) when valid email is submitted", async () => {
    const user = userEvent.setup();
    renderComponent();

    const emailInput = screen.getByPlaceholderText("name@email.com");
    await user.type(emailInput, "user@example.com");

    const verifyButton = screen.getByRole("button", { name: I18nKey.AUTH$VERIFY });
    await user.click(verifyButton);

    expect(mockSetIsReadyToVerifyEmail).toHaveBeenCalledWith(true);
  });

  it("should clear error when valid email is submitted after error", async () => {
    const user = userEvent.setup();
    renderComponent();

    // First submit invalid email
    const emailInput = screen.getByPlaceholderText("name@email.com");
    await user.type(emailInput, "invalid");

    const verifyButton = screen.getByRole("button", { name: I18nKey.AUTH$VERIFY });
    await user.click(verifyButton);

    expect(screen.getByText(I18nKey.AUTH$INVALID_EMAIL_FORMAT)).toBeInTheDocument();

    // Clear and type valid email
    await user.clear(emailInput);
    await user.type(emailInput, "user@example.com");
    await user.click(verifyButton);

    expect(screen.queryByText(I18nKey.AUTH$INVALID_EMAIL_FORMAT)).not.toBeInTheDocument();
    expect(mockSetIsReadyToVerifyEmail).toHaveBeenCalledWith(true);
  });

  it("should submit form when Enter key is pressed", async () => {
    const user = userEvent.setup();
    renderComponent();

    const emailInput = screen.getByPlaceholderText("name@email.com");
    await user.type(emailInput, "user@example.com{enter}");

    expect(mockSetIsReadyToVerifyEmail).toHaveBeenCalledWith(true);
  });
});
