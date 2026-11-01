import React from "react";
import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { BudgetDisplay } from "#/components/features/conversation-panel/budget-display";

describe("BudgetDisplay", () => {
  it("renders nothing when cost is null", () => {
    const { container } = renderWithProviders(
      <BudgetDisplay cost={null} maxBudgetPerTask={5} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when maxBudgetPerTask is null (no budget cap)", () => {
    // Per OHE-3110: there is no way for a user to set max_budget_per_task, so
    // the "No budget limit" line must be omitted rather than shown.
    const { container } = renderWithProviders(
      <BudgetDisplay cost={1.23} maxBudgetPerTask={null} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when maxBudgetPerTask is zero", () => {
    const { container } = renderWithProviders(
      <BudgetDisplay cost={1.23} maxBudgetPerTask={0} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when maxBudgetPerTask is negative", () => {
    const { container } = renderWithProviders(
      <BudgetDisplay cost={1.23} maxBudgetPerTask={-1} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the progress bar and usage text when a real cap exists", () => {
    const { container } = renderWithProviders(
      <BudgetDisplay cost={1.23} maxBudgetPerTask={5} />,
    );
    // Cap-present branch renders two child rows: BudgetProgressBar (whose
    // fill div carries the transition-all class) and BudgetUsageText (the
    // usage key renders untranslated in tests). Asserting on both proves we
    // did not take the cap-absent (null) branch.
    expect(container.querySelector(".transition-all")).toBeInTheDocument();
    expect(
      screen.getByText("CONVERSATION$BUDGET_USAGE_FORMAT"),
    ).toBeInTheDocument();
  });
});
