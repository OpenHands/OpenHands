import { useTranslation } from "react-i18next";
import { BudgetDisplay } from "../../conversation-panel/budget-display";
import { I18nKey } from "#/i18n/declaration";
import { formatUsageCost, isUnpricedUsage } from "#/utils/conversation-metrics";

interface CostSectionProps {
  cost: number | null;
  maxBudget: number | null;
  usageTokens: number | null;
  /** Some model spent tokens without a reported cost, so the total omits it. */
  hasUnpricedModel: boolean;
  /** ACP conversations already carry their own plan-usage estimate note. */
  showEstimateNote: boolean;
}

export function CostSection({
  cost,
  maxBudget,
  usageTokens,
  hasUnpricedModel,
  showEstimateNote,
}: CostSectionProps) {
  const { t } = useTranslation("openhands");

  if (cost === null) {
    return null;
  }

  const tokens = usageTokens ?? 0;
  // Unpriced usage never counts toward the budget, so a 0% bar would mislead.
  const isUnpriced = isUnpricedUsage(cost, tokens);

  return (
    <>
      {!isUnpriced && <BudgetDisplay cost={cost} maxBudget={maxBudget} />}
      <div className="flex justify-between items-center border-t border-divider pt-2">
        <span className="font-semibold">
          {t(I18nKey.CONVERSATION$TOTAL_COST)}
        </span>
        <span className="font-semibold" data-testid="total-cost-value">
          {formatUsageCost(cost, tokens)}
        </span>
      </div>
      {(isUnpriced || hasUnpricedModel) && (
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
