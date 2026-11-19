import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CostSection } from "#/components/features/conversation/metrics-modal/cost-section";
import { I18nKey } from "#/i18n/declaration";

// The global vitest setup mocks useTranslation to return the key itself, so
// these tests assert on raw I18nKey strings.
const renderCost = (
  cost: number | null,
  usageTokens: number | null,
  showEstimateNote = true,
) =>
  render(
    <CostSection
      cost={cost}
      maxBudget={null}
      usageTokens={usageTokens}
      showEstimateNote={showEstimateNote}
    />,
  );

describe("CostSection", () => {
  it("renders nothing when there is no cost data at all", () => {
    const { container } = renderCost(null, null);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the accumulated cost with 4-decimal formatting and an estimate note", () => {
    renderCost(0.003, 1_000);
    expect(
      screen.getByText(I18nKey.CONVERSATION$TOTAL_COST),
    ).toBeInTheDocument();
    expect(screen.getByTestId("total-cost-value")).toHaveTextContent("$0.0030");
    expect(
      screen.getByText(I18nKey.CONVERSATION$COST_ESTIMATE_NOTE),
    ).toBeInTheDocument();
  });

  it("omits the generic estimate note when the caller has its own", () => {
    renderCost(1.25, 1_000, false);
    expect(
      screen.queryByText(I18nKey.CONVERSATION$COST_ESTIMATE_NOTE),
    ).not.toBeInTheDocument();
  });

  it("shows tokens recorded without any cost as unpriced, not $0.0000", () => {
    renderCost(0, 20_162_780);
    expect(screen.getByTestId("total-cost-value")).toHaveTextContent("—");
    expect(screen.queryByText("$0.0000")).not.toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.CONVERSATION$NO_COST_DATA),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(I18nKey.CONVERSATION$COST_ESTIMATE_NOTE),
    ).not.toBeInTheDocument();
  });

  it("keeps $0.0000 when no tokens were used yet", () => {
    renderCost(0, 0);
    expect(screen.getByTestId("total-cost-value")).toHaveTextContent("$0.0000");
    expect(
      screen.queryByText(I18nKey.CONVERSATION$NO_COST_DATA),
    ).not.toBeInTheDocument();
  });
});
