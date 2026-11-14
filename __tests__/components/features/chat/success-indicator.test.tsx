import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SuccessIndicator } from "#/components/features/chat/success-indicator";

describe("SuccessIndicator", () => {
  it("renders an error icon for failed results", () => {
    const { container } = render(<SuccessIndicator status="error" />);
    expect(container.querySelector("svg")).toBeInTheDocument();
  });

  it("renders a clock icon for timeouts", () => {
    const { container } = render(<SuccessIndicator status="timeout" />);
    expect(container.querySelector("svg")).toBeInTheDocument();
  });

  it("renders no icon for successful results", () => {
    const { container } = render(<SuccessIndicator status="success" />);
    expect(container.querySelector("svg")).not.toBeInTheDocument();
  });
});
