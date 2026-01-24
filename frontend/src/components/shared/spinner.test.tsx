import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { Spinner } from "./spinner";

describe("Spinner", () => {
  it("renders a spinner with status role", () => {
    render(<Spinner data-testid="spinner" />);
    const spinner = screen.getByTestId("spinner");
    expect(spinner).toBeInTheDocument();
    expect(spinner).toHaveAttribute("role", "status");
  });

  it("renders an optional label", () => {
    render(<Spinner label="Loading…" />);
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("supports size variants", () => {
    render(<Spinner size="sm" data-testid="spinner-sm" />);
    const wrapper = screen.getByTestId("spinner-sm");
    const ring = wrapper.firstElementChild;
    expect(ring).not.toBeNull();
    expect(ring).toHaveClass("w-4", "h-4");
  });
});
