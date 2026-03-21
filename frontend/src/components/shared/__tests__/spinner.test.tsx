import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Spinner } from "#/components/shared/spinner";

describe("Spinner", () => {
  it("renders with default props", () => {
    render(<Spinner />);
    const spinner = screen.getByTestId("spinner");
    expect(spinner).toBeInTheDocument();
    expect(spinner.querySelector("svg")).toBeInTheDocument();
  });

  it("renders with different sizes", () => {
    const { rerender } = render(<Spinner size="sm" />);
    let spinner = screen.getByTestId("spinner");
    expect(spinner.querySelector("svg")).toHaveClass("w-4", "h-4");

    rerender(<Spinner size="md" />);
    spinner = screen.getByTestId("spinner");
    expect(spinner.querySelector("svg")).toHaveClass("w-6", "h-6");

    rerender(<Spinner size="lg" />);
    spinner = screen.getByTestId("spinner");
    expect(spinner.querySelector("svg")).toHaveClass("w-8", "h-8");

    rerender(<Spinner size="xl" />);
    spinner = screen.getByTestId("spinner");
    expect(spinner.querySelector("svg")).toHaveClass("w-12", "h-12");
  });

  it("renders with label", () => {
    render(<Spinner label="Loading..." />);
    expect(screen.getByText("Loading...")).toBeInTheDocument();
  });

  it("applies custom className", () => {
    render(<Spinner className="custom-class" />);
    const spinner = screen.getByTestId("spinner");
    expect(spinner).toHaveClass("custom-class");
  });

  it("applies custom color style", () => {
    render(<Spinner color="#6366f1" />);
    const svg = screen.getByTestId("spinner").querySelector("svg");
    expect(svg).toHaveStyle({ color: "#6366f1" });
  });

  it("uses custom testId", () => {
    render(<Spinner data-testid="custom-spinner" />);
    expect(screen.getByTestId("custom-spinner")).toBeInTheDocument();
  });
});
