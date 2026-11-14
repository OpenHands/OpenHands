import { useTranslation } from "react-i18next";
import { BudgetDisplay } from "../../conversation-panel/budget-display";
import { I18nKey } from "#/i18n/declaration";

interface CostSectionProps {
  cost: number | null;
  maxBudget: number | null;
  usageTokens: number | null;
  /** ACP conversations already carry their own plan-usage estimate note. */
  showEstimateNote: boolean;
}

export function CostSection({
  cost,
  maxBudget,
  usageTokens,
  showEstimateNote,
}: CostSectionProps) {
  const { t } = useTranslation("openhands");

  if (cost === null) {
    return null;
  }

  // Calls the agent server cannot price are recorded at cost 0 while their
  // tokens still count, so "$0.0000" here would read as "this was free".
  const isUnpriced = cost === 0 && (usageTokens ?? 0) > 0;

  return (
    <>
      <BudgetDisplay cost={cost} maxBudget={maxBudget} />
      <div className="flex justify-between items-center border-t border-divider pt-2">
        <span className="font-semibold">
          {t(I18nKey.CONVERSATION$TOTAL_COST)}
        </span>
        <span className="font-semibold" data-testid="total-cost-value">
          {isUnpriced ? "—" : `$${cost.toFixed(4)}`}
        </span>
      </div>
      {isUnpriced && (
        <span className="text-xs text-muted" data-testid="cost-unpriced-note">
          {t(I18nKey.CONVERSATION$NO_COST_DATA)}
        </span>
      )}
      {!isUnpriced && showEstimateNote && (
        <span className="text-xs text-muted" data-testid="cost-estimate-note">
          {t(I18nKey.CONVERSATION$COST_ESTIMATE_NOTE)}
        </span>
      )}
    </>
  );
}
