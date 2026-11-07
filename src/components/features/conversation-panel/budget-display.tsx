import { BudgetProgressBar } from "./budget-progress-bar";
import { BudgetUsageText } from "./budget-usage-text";

interface BudgetDisplayProps {
  cost: number | null;
  maxBudgetPerTask: number | null;
}

export function BudgetDisplay({ cost, maxBudgetPerTask }: BudgetDisplayProps) {
  // Only render the cap/progress UI when a real per-conversation budget cap
  // exists. There is currently no way for a user to set max_budget_per_task,
  // so showing a "No budget limit" line is misleading — omit it entirely
  // and let the surrounding cost/usage rows speak for themselves.
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
