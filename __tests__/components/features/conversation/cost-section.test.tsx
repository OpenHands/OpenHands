import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { renderWithProviders } from "test-utils";
import { CostSection } from "#/components/features/conversation/metrics-modal/cost-section";

describe("CostSection", () => {
  it.each([null, 0])(
    "shows total cost without a no-budget fallback for %s max budget",
    (maxBudgetPerTask) => {
      renderWithProviders(
        <CostSection cost={1.25} maxBudgetPerTask={maxBudgetPerTask} />,
      );

      expect(screen.getByText("CONVERSATION$TOTAL_COST")).toBeInTheDocument();
      expect(screen.getByText("$1.2500")).toBeInTheDocument();
      expect(
        screen.queryByText("CONVERSATION$NO_BUDGET_LIMIT"),
      ).not.toBeInTheDocument();
    },
  );

  it("keeps budget usage visible when a positive limit exists", () => {
    renderWithProviders(<CostSection cost={1.25} maxBudgetPerTask={10} />);

    expect(
      screen.getByText("CONVERSATION$BUDGET_USAGE_FORMAT"),
    ).toBeInTheDocument();
  });
});
