import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ErrorMessage } from "#/components/features/chat/error-message";

describe("ErrorMessage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the error message", () => {
    render(<ErrorMessage defaultMessage="LLM connection timed out" />);

    expect(screen.getByText("CHAT_INTERFACE$AGENT_ERROR_MESSAGE")).toBeInTheDocument();
    expect(screen.queryByTestId("error-message-retry")).not.toBeInTheDocument();
  });

  it("renders retry button when onRetry is provided", () => {
    const onRetry = vi.fn();
    render(
      <ErrorMessage
        defaultMessage="Rate limit reached"
        onRetry={onRetry}
      />,
    );

    expect(screen.getByTestId("error-message-retry")).toBeInTheDocument();
    expect(screen.getByTestId("error-message-retry")).not.toBeDisabled();
  });

  it("calls onRetry when retry button is clicked", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();

    render(
      <ErrorMessage
        defaultMessage="Server 500 error"
        onRetry={onRetry}
      />,
    );

    await user.click(screen.getByTestId("error-message-retry"));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("disables retry button when isRetrying is true", () => {
    const onRetry = vi.fn();

    render(
      <ErrorMessage
        defaultMessage="Server 500 error"
        onRetry={onRetry}
        isRetrying
      />,
    );

    expect(screen.getByTestId("error-message-retry")).toBeDisabled();
  });

  it("expands details when toggle button is clicked", async () => {
    const user = userEvent.setup();
    render(<ErrorMessage defaultMessage="Detailed stack trace" />);

    expect(screen.queryByText("Detailed stack trace")).not.toBeInTheDocument();

    const toggle = screen.getByRole("button", {
      name: /COMMON\$VIEW_MORE/i,
    });
    await user.click(toggle);

    expect(screen.getByText("Detailed stack trace")).toBeInTheDocument();
  });
});
