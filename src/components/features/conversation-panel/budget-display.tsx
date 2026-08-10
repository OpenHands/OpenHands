import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { BudgetProgressBar } from "./budget-progress-bar";
import { BudgetUsageText } from "./budget-usage-text";

interface BudgetDisplayProps {
  cost: number | null;
  maxBudgetPerTask: number | null;
}

export function BudgetDisplay({ cost, maxBudgetPerTask }: BudgetDisplayProps) {
  const { t } = useTranslation("openhands");

  // Don't render anything if cost is not available
  if (cost === null) {
    return null;
  }

  return (
    <div className="border-b border-[var(--oh-border-subtle)]">
      {maxBudgetPerTask !== null && maxBudgetPerTask > 0 ? (
        <>
          <BudgetProgressBar currentCost={cost} maxBudget={maxBudgetPerTask} />
          <BudgetUsageText currentCost={cost} maxBudget={maxBudgetPerTask} />
        </>
      ) : (
        <span className="text-xs text-[var(--oh-muted)]">
          {t(I18nKey.CONVERSATION$NO_BUDGET_LIMIT)}
        </span>
      )}
    </div>
  );
}
