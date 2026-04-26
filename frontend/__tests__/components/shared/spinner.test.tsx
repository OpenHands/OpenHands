import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Spinner } from "#/components/shared/spinner";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => (key === "HOME$LOADING" ? "Loading" : key),
  }),
}));

describe("Spinner", () => {
  it("renders an accessible loading state without a label", () => {
    render(<Spinner testId="spinner" />);

    expect(screen.getByTestId("spinner")).toHaveAttribute("role", "status");
    expect(screen.getByText("Loading")).toHaveClass("sr-only");
  });

  it("renders the provided label", () => {
    render(<Spinner label="Loading repositories" />);

    expect(screen.getByText("Loading repositories")).toBeInTheDocument();
  });

  it("applies the requested size variant", () => {
    render(<Spinner size="xl" testId="spinner" />);

    const spinnerCircle = screen.getByTestId("spinner").querySelector("div");

    expect(spinnerCircle).toHaveClass("h-16", "w-16", "border-4");
  });
});
