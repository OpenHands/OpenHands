import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

interface BudgetUsageTextProps {
  currentCost: number;
  maxBudget: number;
}

export function BudgetUsageText({
  currentCost,
  maxBudget,
}: BudgetUsageTextProps) {
  const { t } = useTranslation("openhands");
  const usagePercentage = (currentCost / maxBudget) * 100;

  return (
    <div className="flex justify-end">
      <span className="text-xs text-[var(--oh-muted)]">
        {t(I18nKey.CONVERSATION$BUDGET_USAGE_FORMAT, {
          currentCost: `$${currentCost.toFixed(4)}`,
          maxBudget: `$${maxBudget.toFixed(4)}`,
          usagePercentage: usagePercentage.toFixed(2),
          used: t(I18nKey.CONVERSATION$USED),
        })}
      </span>
    </div>
  );
}
