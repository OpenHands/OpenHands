import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { Spinner } from "./spinner";

describe("Spinner", () => {
  it("renders a spinner with status role", () => {
    render(<Spinner testId="spinner" />);
    const wrapper = screen.getByTestId("spinner");
    expect(wrapper).toBeInTheDocument();
    expect(wrapper).toHaveAttribute("role", "status");
  });

  it("renders an optional label", () => {
    render(<Spinner label="Loading…" />);
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("supports size variants", () => {
    render(<Spinner size="sm" testId="spinner-sm" />);
    const wrapper = screen.getByTestId("spinner-sm");
    const ring = wrapper.firstElementChild;
    expect(ring).not.toBeNull();
    expect(ring).toHaveClass("w-4", "h-4");
  });
});
