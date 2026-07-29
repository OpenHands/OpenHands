import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ErrorMessageBanner } from "#/components/features/chat/error-message-banner";

const toastMocks = vi.hoisted(() => ({
  displayErrorToast: vi.fn(),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: toastMocks.displayErrorToast,
}));

describe("ErrorMessageBanner", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    toastMocks.displayErrorToast.mockClear();
  });

  it("calls onDismiss when the close button is clicked", async () => {
    const user = userEvent.setup();
    const onDismiss = vi.fn();

    render(
      <ErrorMessageBanner
        message="Something went wrong"
        onDismiss={onDismiss}
      />,
    );

    await user.click(screen.getByTestId("error-message-banner-dismiss"));
    expect(onDismiss).toHaveBeenCalledTimes(1);
  });

  it("calls onRetry when the retry button is clicked", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();

    render(
      <ErrorMessageBanner
        message="Unable to connect to server"
        onRetry={onRetry}
      />,
    );

    await user.click(screen.getByTestId("error-message-banner-retry"));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("shows a red error icon beside the message", () => {
    render(<ErrorMessageBanner message="Something went wrong" />);

    const icon = screen.getByTestId("error-message-banner-icon");
    expect(icon).toHaveStyle({ color: "var(--oh-status-error)" });
  });

  it("uses greyscale theme tokens instead of red error styling", () => {
    render(<ErrorMessageBanner message="Something went wrong" />);

    const banner = screen.getByTestId("error-message-banner");
    expect(banner.className).toContain("border-[var(--oh-border)]");
    expect(banner.className).toContain("bg-[var(--oh-surface-raised)]");
    expect(banner.className).not.toContain("#FF0006");
    expect(banner.className).not.toContain("#4A0709");
  });

  it("shows a View More / View Less toggle for long messages", async () => {
    const user = userEvent.setup();
    const longMessage = "a".repeat(400);

    render(<ErrorMessageBanner message={longMessage} />);

    const toggle = screen.getByTestId("error-message-banner-toggle");
    expect(toggle).toHaveTextContent("COMMON$VIEW_MORE");

    await user.click(toggle);
    expect(toggle).toHaveTextContent("COMMON$VIEW_LESS");
  });

  it("copies the full error message when the copy button is clicked", async () => {
    const user = userEvent.setup();
    const longMessage = `first line\n${"long error detail ".repeat(40)}`;
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.spyOn(navigator.clipboard, "writeText").mockImplementation(writeText);

    render(<ErrorMessageBanner message={longMessage} onDismiss={vi.fn()} />);

    await user.click(screen.getByTestId("error-message-banner-copy"));

    expect(writeText).toHaveBeenCalledWith(longMessage);
    await waitFor(() =>
      expect(
        screen.getByTestId("error-message-banner-copy"),
      ).toHaveAccessibleName("BUTTON$COPIED"),
    );
  });

  it("shows an error toast when the copy button cannot write to the clipboard", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockRejectedValue(new Error("denied"));
    vi.spyOn(navigator.clipboard, "writeText").mockImplementation(writeText);

    render(<ErrorMessageBanner message="copy failed" />);

    await user.click(screen.getByTestId("error-message-banner-copy"));

    expect(writeText).toHaveBeenCalledWith("copy failed");
    expect(toastMocks.displayErrorToast).toHaveBeenCalledWith(
      "CHAT_INTERFACE$CHAT_MESSAGE_COPY_FAILED",
    );
    expect(
      screen.getByTestId("error-message-banner-copy"),
    ).toHaveAccessibleName("BUTTON$COPY");
  });

  it("renders a code-specific header for a known ACP error code", () => {
    render(
      <ErrorMessageBanner
        message="[-32603] Internal error: 401 invalid x-api-key"
        code="ACPAuthRequired"
      />,
    );

    const header = screen.getByTestId("error-message-banner-header");
    expect(header).toHaveTextContent("ERROR$ACP_AUTH_REQUIRED_TITLE");
    // The detail is still shown verbatim below the header.
    expect(
      screen.getByTestId("error-message-banner-content"),
    ).toHaveTextContent("invalid x-api-key");
  });

  it("renders no header for an unknown or absent code", () => {
    render(<ErrorMessageBanner message="boom" code={null} />);
    expect(
      screen.queryByTestId("error-message-banner-header"),
    ).not.toBeInTheDocument();
  });

  it("shows a re-auth action and calls onReauth when provided", async () => {
    const user = userEvent.setup();
    const onReauth = vi.fn();

    render(
      <ErrorMessageBanner
        message="Authentication failed"
        code="ACPAuthRequired"
        onReauth={onReauth}
      />,
    );

    const reauth = screen.getByTestId("error-message-banner-reauth");
    expect(reauth).toHaveTextContent("ERROR$ACP_UPDATE_CREDENTIALS");
    await user.click(reauth);
    expect(onReauth).toHaveBeenCalledTimes(1);
  });

  it("omits the re-auth action when onReauth is not provided", () => {
    render(<ErrorMessageBanner message="boom" code="ACPPromptError" />);
    expect(
      screen.queryByTestId("error-message-banner-reauth"),
    ).not.toBeInTheDocument();
  });
});
