import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CostSection } from "#/components/features/conversation/metrics-modal/cost-section";
import { I18nKey } from "#/i18n/declaration";

// The global vitest setup mocks useTranslation to return the key itself, so
// these tests assert on raw I18nKey strings.
describe("CostSection", () => {
  it("renders nothing when there is no cost data at all", () => {
    const { container } = render(
      <CostSection cost={null} maxBudgetPerTask={null} usageTokens={null} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the accumulated cost with 4-decimal formatting", () => {
    render(
      <CostSection cost={0.003} maxBudgetPerTask={null} usageTokens={null} />,
    );
    expect(
      screen.getByText(I18nKey.CONVERSATION$TOTAL_COST),
    ).toBeInTheDocument();
    expect(screen.getByText("$0.0030")).toBeInTheDocument();
  });

  it("explains that the total is an estimate", () => {
    render(
      <CostSection cost={1.25} maxBudgetPerTask={null} usageTokens={null} />,
    );
    expect(
      screen.getByText(I18nKey.CONVERSATION$COST_ESTIMATE_NOTE),
    ).toBeInTheDocument();
  });

  it("flags tokens recorded without any cost as no-cost-data, not free", () => {
    render(
      <CostSection cost={0} maxBudgetPerTask={null} usageTokens={20_162_780} />,
    );
    expect(
      screen.getByText(I18nKey.CONVERSATION$NO_COST_DATA),
    ).toBeInTheDocument();
  });

  it("does not flag the no-cost-data note when no tokens were used", () => {
    render(<CostSection cost={0} maxBudgetPerTask={null} usageTokens={0} />);
    expect(
      screen.queryByText(I18nKey.CONVERSATION$NO_COST_DATA),
    ).not.toBeInTheDocument();
  });

  it("does not flag the no-cost-data note when a cost was recorded", () => {
    render(<CostSection cost={0.5} maxBudgetPerTask={null} usageTokens={99} />);
    expect(
      screen.queryByText(I18nKey.CONVERSATION$NO_COST_DATA),
    ).not.toBeInTheDocument();
  });
});
