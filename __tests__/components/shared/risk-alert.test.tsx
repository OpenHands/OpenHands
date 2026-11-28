import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RiskAlert } from "#/components/shared/risk-alert";

describe("RiskAlert", () => {
  it("renders a high-severity alert with title and content", () => {
    render(
      <RiskAlert
        severity="high"
        title="High Risk"
        content="Review carefully."
      />,
    );

    expect(screen.getByText("High Risk")).toBeInTheDocument();
    expect(screen.getByText("Review carefully.")).toBeInTheDocument();
  });

  it("renders a medium-severity alert with title and content", () => {
    render(
      <RiskAlert
        severity="medium"
        title="Medium Risk"
        content="May have side effects."
      />,
    );

    expect(screen.getByText("Medium Risk")).toBeInTheDocument();
    expect(screen.getByText("May have side effects.")).toBeInTheDocument();
  });

  it("renders a custom icon when provided", () => {
    render(
      <RiskAlert
        severity="high"
        title="High Risk"
        content="Check before proceeding."
        icon={<svg data-testid="custom-icon" />}
      />,
    );

    expect(screen.getByTestId("custom-icon")).toBeInTheDocument();
  });

  it("does not render for low severity", () => {
    const { container } = render(
      <RiskAlert severity="low" title="Low Risk" content="Minimal impact." />,
    );

    expect(container.firstChild).toBeNull();
  });

  it("applies additional className to the alert container", () => {
    render(
      <RiskAlert
        severity="medium"
        title="Medium Risk"
        content="Needs review."
        className="extra-class"
      />,
    );

    const container = screen.getByText("Medium Risk").closest("div");
    expect(container).toHaveClass("extra-class");
  });
});
