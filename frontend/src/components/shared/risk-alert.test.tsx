import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RiskAlert } from "./risk-alert";

describe("RiskAlert", () => {
  it("renders the title and content", () => {
    render(
      <RiskAlert
        severity="high"
        title="High severity title"
        content="High severity content"
      />,
    );

    expect(screen.getByText("High severity title")).toBeInTheDocument();
    expect(screen.getByText("High severity content")).toBeInTheDocument();
  });

  it("renders the icon when provided", () => {
    render(
      <RiskAlert
        severity="high"
        title="t"
        content="c"
        icon={<span data-testid="alert-icon">!</span>}
      />,
    );

    expect(screen.getByTestId("alert-icon")).toBeInTheDocument();
  });

  it("applies high severity styling (red border and red icon)", () => {
    const { container } = render(
      <RiskAlert
        severity="high"
        title="t"
        content="c"
        icon={<span data-testid="alert-icon">!</span>}
      />,
    );

    const root = container.firstChild as HTMLElement;
    expect(root.className).toContain("border-[#FF0006]");
    expect(root.className).toContain("bg-[#4A0709]");

    const icon = screen.getByTestId("alert-icon").parentElement;
    expect(icon?.className).toContain("text-red-400");
  });

  it("applies medium severity styling (amber border and amber icon)", () => {
    const { container } = render(
      <RiskAlert
        severity="medium"
        title="Medium severity title"
        content="Medium severity content"
        icon={<span data-testid="alert-icon">!</span>}
      />,
    );

    const root = container.firstChild as HTMLElement;
    expect(root.className).toContain("border-amber-500");
    expect(root.className).toContain("bg-[#3a2a06]");

    const icon = screen.getByTestId("alert-icon").parentElement;
    expect(icon?.className).toContain("text-amber-400");

    expect(screen.getByText("Medium severity title")).toBeInTheDocument();
    expect(screen.getByText("Medium severity content")).toBeInTheDocument();
  });
});
