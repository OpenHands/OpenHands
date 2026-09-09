import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useFormatCost } from "#/stores/cost-currency-store";

interface BudgetUsageTextProps {
  currentCost: number;
  maxBudget: number;
}

export function BudgetUsageText({
  currentCost,
  maxBudget,
}: BudgetUsageTextProps) {
  const { t } = useTranslation("openhands");
  const formatCost = useFormatCost();
  const usagePercentage = (currentCost / maxBudget) * 100;

  return (
    <div className="flex justify-end">
      <span className="text-xs text-[var(--oh-muted)]">
        {t(I18nKey.CONVERSATION$BUDGET_USAGE_FORMAT, {
          currentCost: formatCost(currentCost, undefined, { detailed: true }),
          maxBudget: formatCost(maxBudget, undefined, { detailed: true }),
          usagePercentage: usagePercentage.toFixed(2),
          used: t(I18nKey.CONVERSATION$USED),
        })}
      </span>
    </div>
  );
}
