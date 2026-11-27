import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi, describe, it, expect } from "vitest";
import { ErrorState } from "#/components/features/automations/error-state";

describe("ErrorState", () => {
  it("renders error icon and retry button", () => {
    const { container } = render(<ErrorState onRetry={vi.fn()} />);

    // Check for the error icon (ExclamationCircleIcon)
    expect(container.querySelector("svg")).toBeInTheDocument();
    // Check for the retry button
    expect(screen.getByRole("button")).toBeInTheDocument();
  });

  it("calls onRetry when retry button is clicked", async () => {
    const onRetry = vi.fn();
    const user = userEvent.setup();
    render(<ErrorState onRetry={onRetry} />);

    await user.click(screen.getByRole("button"));

    expect(onRetry).toHaveBeenCalledOnce();
  });
});
