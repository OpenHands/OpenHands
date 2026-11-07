import { useTranslation } from "react-i18next";
import { BudgetDisplay } from "../../conversation-panel/budget-display";
import { I18nKey } from "#/i18n/declaration";

interface CostSectionProps {
  cost: number | null;
  maxBudgetPerTask: number | null;
  usageTokens: number | null;
}

export function CostSection({
  cost,
  maxBudgetPerTask,
  usageTokens,
}: CostSectionProps) {
  const { t } = useTranslation("openhands");

  if (cost === null) {
    return null;
  }

  // The agent server records cost only for calls it could price; tokens
  // without a matching cost mean the model's pricing was unknown, so a
  // flat "$0.0000" would misread as "this session was free".
  const hasTokensWithoutCost = cost === 0 && (usageTokens ?? 0) > 0;

  return (
    <>
      <BudgetDisplay cost={cost} maxBudgetPerTask={maxBudgetPerTask} />
      <div className="flex justify-between items-center border-t border-border-subtle pt-2">
        <span className="font-semibold">
          {t(I18nKey.CONVERSATION$TOTAL_COST)}
        </span>
        <span className="font-semibold">${cost.toFixed(4)}</span>
      </div>
      {hasTokensWithoutCost && (
        <span className="text-xs text-muted">
          {t(I18nKey.CONVERSATION$NO_COST_DATA)}
        </span>
      )}
      <span className="text-xs text-muted">
        {t(I18nKey.CONVERSATION$COST_ESTIMATE_NOTE)}
      </span>
    </>
  );
}
