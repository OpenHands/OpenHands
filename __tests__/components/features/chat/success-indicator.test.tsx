import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { SuccessIndicator } from "#/components/features/chat/success-indicator";

describe("SuccessIndicator", () => {
  it("renders no icon for a successful result", () => {
    render(<SuccessIndicator status="success" />);

    expect(screen.queryByTestId("status-icon")).not.toBeInTheDocument();
  });

  it("renders a clock icon for a timed-out result", () => {
    render(<SuccessIndicator status="timeout" />);

    expect(screen.getByTestId("status-icon")).toHaveClass("fill-yellow-500");
  });

  it("renders a failure icon for an errored result", () => {
    render(<SuccessIndicator status="error" />);

    expect(screen.getByTestId("status-icon")).toHaveClass(
      "fill-status-fail-solid",
    );
  });
});
