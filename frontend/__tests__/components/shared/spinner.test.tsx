import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Spinner } from "#/components/shared/spinner";

describe("Spinner", () => {
  it("renders medium spinner by default", () => {
    render(<Spinner />);

    const icon = screen.getByTestId("spinner");

    expect(icon).toBeInTheDocument();
    expect(icon).toHaveClass("animate-spin");
    expect(icon).toHaveClass("h-6");
    expect(icon).toHaveClass("w-6");
  });

  it("renders label and xl size when provided", () => {
    render(
      <Spinner
        size="xl"
        label="Loading data"
        className="text-white"
        testId="xl-spinner"
      />,
    );

    const icon = screen.getByTestId("xl-spinner");

    expect(icon).toHaveClass("h-16");
    expect(icon).toHaveClass("w-16");
    expect(icon).toHaveClass("text-white");
    expect(screen.getByText("Loading data")).toBeInTheDocument();
  });
});
