import { BudgetProgressBar } from "./budget-progress-bar";
import { BudgetUsageText } from "./budget-usage-text";

interface BudgetDisplayProps {
  cost: number | null;
  maxBudgetPerTask: number | null;
}

export function BudgetDisplay({ cost, maxBudgetPerTask }: BudgetDisplayProps) {
  if (cost === null || maxBudgetPerTask === null || maxBudgetPerTask <= 0) {
    return null;
  }

  return (
    <div>
      <BudgetProgressBar currentCost={cost} maxBudget={maxBudgetPerTask} />
      <BudgetUsageText currentCost={cost} maxBudget={maxBudgetPerTask} />
    </div>
  );
}
