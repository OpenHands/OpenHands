import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Card } from "#/ui/card";

describe("Card", () => {
  it("should render children", () => {
    render(<Card>Card Content</Card>);

    expect(screen.getByText("Card Content")).toBeInTheDocument();
  });

  it("should render with testId", () => {
    render(<Card testId="test-card">Content</Card>);

    expect(screen.getByTestId("test-card")).toBeInTheDocument();
  });

  // Prop-passthrough contract: consumer-supplied className must land on the
  // element.
  it("should apply custom className", () => {
    render(
      <Card testId="test-card" className="custom-class">
        Content
      </Card>,
    );

    expect(screen.getByTestId("test-card")).toHaveClass("custom-class");
  });
});
