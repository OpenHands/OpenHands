import { useTranslation } from "react-i18next";
import { BudgetDisplay } from "../../conversation-panel/budget-display";
import { I18nKey } from "#/i18n/declaration";

interface CostSectionProps {
  cost: number | null;
  maxBudgetPerTask: number | null;
}

export function CostSection({ cost, maxBudgetPerTask }: CostSectionProps) {
  const { t } = useTranslation("openhands");

  if (cost === null) {
    return null;
  }

  return (
    <>
      <BudgetDisplay cost={cost} maxBudgetPerTask={maxBudgetPerTask} />
      <div className="flex justify-between items-center border-t border-[var(--oh-border-subtle)] pt-2">
        <span className="font-semibold">
          {t(I18nKey.CONVERSATION$TOTAL_COST)}
        </span>
        <span className="font-semibold">${cost.toFixed(4)}</span>
      </div>
    </>
  );
}
