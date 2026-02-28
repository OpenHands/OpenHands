import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import StepHeader from "#/components/features/onboarding/step-header";

describe("StepHeader", () => {
  const defaultProps = {
    title: "Test Title",
    currentStep: 1,
    totalSteps: 3,
  };

  it("should render with the correct test id", () => {
    render(<StepHeader {...defaultProps} />);

    expect(screen.getByTestId("step-header")).toBeInTheDocument();
  });

  it("should display the title", () => {
    render(<StepHeader {...defaultProps} />);

    expect(screen.getByText("Test Title")).toBeInTheDocument();
  });

  it("should render correct number of progress dots based on totalSteps", () => {
    render(<StepHeader {...defaultProps} totalSteps={5} />);

    const stepHeader = screen.getByTestId("step-header");
    const progressDots = stepHeader.querySelectorAll(".rounded-full");
    expect(progressDots).toHaveLength(5);
  });

  it("should fill progress dots up to currentStep", () => {
    render(<StepHeader {...defaultProps} currentStep={2} totalSteps={4} />);

    const stepHeader = screen.getByTestId("step-header");
    const filledDots = stepHeader.querySelectorAll(".bg-white");
    const unfilledDots = stepHeader.querySelectorAll(".bg-neutral-600");

    expect(filledDots).toHaveLength(2);
    expect(unfilledDots).toHaveLength(2);
  });

  it("should show all dots filled when on last step", () => {
    render(<StepHeader {...defaultProps} currentStep={3} totalSteps={3} />);

    const stepHeader = screen.getByTestId("step-header");
    const filledDots = stepHeader.querySelectorAll(".bg-white");
    const unfilledDots = stepHeader.querySelectorAll(".bg-neutral-600");

    expect(filledDots).toHaveLength(3);
    expect(unfilledDots).toHaveLength(0);
  });

  it("should show no dots filled when currentStep is 0", () => {
    render(<StepHeader {...defaultProps} currentStep={0} totalSteps={3} />);

    const stepHeader = screen.getByTestId("step-header");
    const filledDots = stepHeader.querySelectorAll(".bg-white");
    const unfilledDots = stepHeader.querySelectorAll(".bg-neutral-600");

    expect(filledDots).toHaveLength(0);
    expect(unfilledDots).toHaveLength(3);
  });

  it("should handle single step progress", () => {
    render(<StepHeader {...defaultProps} currentStep={1} totalSteps={1} />);

    const stepHeader = screen.getByTestId("step-header");
    const progressDots = stepHeader.querySelectorAll(".rounded-full");
    const filledDots = stepHeader.querySelectorAll(".bg-white");

    expect(progressDots).toHaveLength(1);
    expect(filledDots).toHaveLength(1);
  });
});
