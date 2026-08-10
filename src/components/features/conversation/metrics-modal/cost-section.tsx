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
      <div className="flex justify-between items-center pb-2">
        <span className="text-lg font-semibold">
          {t(I18nKey.CONVERSATION$TOTAL_COST)}
        </span>
        <span className="font-semibold">${cost.toFixed(4)}</span>
      </div>
      <BudgetDisplay cost={cost} maxBudgetPerTask={maxBudgetPerTask} />
    </>
  );
}
