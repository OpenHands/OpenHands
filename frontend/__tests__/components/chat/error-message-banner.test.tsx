import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ErrorMessageBanner } from "#/components/features/chat/error-message-banner";

describe("ErrorMessageBanner", () => {
  it("calls onDismiss when the close button is clicked", async () => {
    const user = userEvent.setup();
    const onDismiss = vi.fn();

    render(
      <ErrorMessageBanner
        message="Something went wrong"
        onDismiss={onDismiss}
      />,
    );

    await user.click(screen.getByLabelText("BUTTON$CLOSE"));
    expect(onDismiss).toHaveBeenCalledTimes(1);
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
});
