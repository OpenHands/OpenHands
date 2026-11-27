import { BudgetProgressBar } from "./budget-progress-bar";
import { BudgetUsageText } from "./budget-usage-text";

interface BudgetDisplayProps {
  cost: number | null;
  maxBudget: number | null;
}

export function BudgetDisplay({ cost, maxBudget }: BudgetDisplayProps) {
  if (cost === null || maxBudget === null || maxBudget <= 0) {
    return null;
  }

  return (
    <div>
      <BudgetProgressBar currentCost={cost} maxBudget={maxBudget} />
      <BudgetUsageText currentCost={cost} maxBudget={maxBudget} />
    </div>
  );
}
