import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import VerifyEmailOTP from "#/components/features/auth/otp-login/verify-email-otp";
import { I18nKey } from "#/i18n/declaration";

describe("VerifyEmailOTP", () => {
  const mockSetIsOTPComplete = vi.fn();
  const mockSetIsReadyToVerifyEmail = vi.fn();

  const renderComponent = () =>
    render(
      <VerifyEmailOTP
        setIsOTPComplete={mockSetIsOTPComplete}
        setIsReadyToVerifyEmail={mockSetIsReadyToVerifyEmail}
      />,
    );

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render heading and instructions", () => {
    renderComponent();

    expect(screen.getByText(I18nKey.AUTH$ENTER_CODE)).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.AUTH$OTP_SENT_MESSAGE),
    ).toBeInTheDocument();
  });

  it("should render 6 OTP input fields", () => {
    renderComponent();

    const inputs = screen.getAllByRole("textbox");
    expect(inputs).toHaveLength(6);
  });

  it("should render Next and Back buttons", () => {
    renderComponent();

    expect(screen.getByRole("button", { name: I18nKey.AUTH$NEXT })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: I18nKey.AUTH$BACK })).toBeInTheDocument();
  });

  it("should call setIsReadyToVerifyEmail(false) when Back button is clicked", async () => {
    const user = userEvent.setup();
    renderComponent();

    const backButton = screen.getByRole("button", { name: I18nKey.AUTH$BACK });
    await user.click(backButton);

    expect(mockSetIsReadyToVerifyEmail).toHaveBeenCalledWith(false);
  });

  it("should only accept numeric input", async () => {
    const user = userEvent.setup();
    renderComponent();

    const inputs = screen.getAllByRole("textbox");

    // Try typing a letter - should not change value
    await user.type(inputs[0], "a");
    expect(inputs[0]).toHaveValue("");

    // Try typing a number - should work
    await user.type(inputs[0], "1");
    expect(inputs[0]).toHaveValue("1");
  });

  it("should auto-focus next input when digit is entered", async () => {
    const user = userEvent.setup();
    renderComponent();

    const inputs = screen.getAllByRole("textbox");

    await user.type(inputs[0], "1");
    expect(inputs[1]).toHaveFocus();

    await user.type(inputs[1], "2");
    expect(inputs[2]).toHaveFocus();
  });

  it("should call setIsOTPComplete(true) when valid OTP is submitted", async () => {
    const user = userEvent.setup();
    renderComponent();

    const inputs = screen.getAllByRole("textbox");

    // Type full OTP
    await user.type(inputs[0], "1");
    await user.type(inputs[1], "2");
    await user.type(inputs[2], "3");
    await user.type(inputs[3], "4");
    await user.type(inputs[4], "5");
    await user.type(inputs[5], "6");

    const nextButton = screen.getByRole("button", { name: I18nKey.AUTH$NEXT });
    await user.click(nextButton);

    expect(mockSetIsOTPComplete).toHaveBeenCalledWith(true);
  });

  it("should allow only single digit per input", async () => {
    const user = userEvent.setup();
    renderComponent();

    const inputs = screen.getAllByRole("textbox");

    // Input should have maxLength=1
    expect(inputs[0]).toHaveAttribute("maxLength", "1");
  });
});
