import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { OnboardingProgressBar } from "#/components/features/onboarding/onboarding-progress-bar";

describe("OnboardingProgressBar", () => {
  it("renders one segment per step and marks the current segment", () => {
    render(<OnboardingProgressBar currentStep={1} totalSteps={4} />);

    const bar = screen.getByTestId("onboarding-progress-bar");
    expect(bar).toHaveAttribute("aria-valuenow", "2");
    expect(bar).toHaveAttribute("aria-valuemin", "1");
    expect(bar).toHaveAttribute("aria-valuemax", "4");

    // Segment states match the current step.
    expect(
      screen.getByTestId("onboarding-progress-step-0"),
    ).toHaveAttribute("data-state", "completed");
    expect(
      screen.getByTestId("onboarding-progress-step-1"),
    ).toHaveAttribute("data-state", "current");
    expect(
      screen.getByTestId("onboarding-progress-step-2"),
    ).toHaveAttribute("data-state", "upcoming");
    expect(
      screen.getByTestId("onboarding-progress-step-3"),
    ).toHaveAttribute("data-state", "upcoming");
  });

  it("treats the first step as current with no completed segments", () => {
    render(<OnboardingProgressBar currentStep={0} totalSteps={3} />);

    expect(
      screen.getByTestId("onboarding-progress-step-0"),
    ).toHaveAttribute("data-state", "current");
    expect(
      screen.getByTestId("onboarding-progress-step-1"),
    ).toHaveAttribute("data-state", "upcoming");
    expect(
      screen.getByTestId("onboarding-progress-step-2"),
    ).toHaveAttribute("data-state", "upcoming");
  });
});
