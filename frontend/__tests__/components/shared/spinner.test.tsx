import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Spinner } from "#/components/shared/spinner";

describe("Spinner", () => {
  it("should render with default size (md)", () => {
    render(<Spinner />);
    const spinner = screen.getByTestId("spinner");
    expect(spinner).toBeInTheDocument();
    expect(spinner).toHaveClass("h-6", "w-6");
  });

  it("should render with sm size", () => {
    render(<Spinner size="sm" />);
    const spinner = screen.getByTestId("spinner");
    expect(spinner).toHaveClass("h-4", "w-4");
  });

  it("should render with lg size", () => {
    render(<Spinner size="lg" />);
    const spinner = screen.getByTestId("spinner");
    expect(spinner).toHaveClass("h-8", "w-8");
  });

  it("should render with xl size", () => {
    render(<Spinner size="xl" />);
    const spinner = screen.getByTestId("spinner");
    expect(spinner).toHaveClass("h-16", "w-16");
  });

  it("should render with a label", () => {
    render(<Spinner label="Loading data..." />);
    expect(screen.getByText("Loading data...")).toBeInTheDocument();
  });

  it("should apply custom className to the spinner element", () => {
    render(<Spinner className="text-white" />);
    const spinner = screen.getByTestId("spinner");
    expect(spinner).toHaveClass("text-white");
  });

  it("should apply custom labelClassName to the label", () => {
    render(<Spinner label="Loading" labelClassName="text-2xl" />);
    const label = screen.getByText("Loading");
    expect(label).toHaveClass("text-2xl");
  });

  it("should have role=status for accessibility", () => {
    render(<Spinner />);
    const spinner = screen.getByRole("status");
    expect(spinner).toBeInTheDocument();
  });

  it("should use label as aria-label when provided", () => {
    render(<Spinner label="Loading items" />);
    const spinner = screen.getByRole("status");
    expect(spinner).toHaveAttribute("aria-label", "Loading items");
  });

  it("should default aria-label to Loading when no label provided", () => {
    render(<Spinner />);
    const spinner = screen.getByRole("status");
    expect(spinner).toHaveAttribute("aria-label", "Loading");
  });
});
