import { cn } from "#/utils/utils";

interface BudgetProgressBarProps {
  currentCost: number;
  maxBudget: number;
}

export function BudgetProgressBar({
  currentCost,
  maxBudget,
}: BudgetProgressBarProps) {
  const usagePercentage = (currentCost / maxBudget) * 100;
  const isNearLimit = usagePercentage > 80;

  return (
    <div className="relative mt-1 h-1.5 w-full rounded-full bg-tertiary">
      <div
        className={cn(
          "absolute inset-y-0 left-0 rounded-full transition-all duration-300",
          isNearLimit ? "bg-red-500" : "bg-blue-500",
        )}
        // runtime usage-percentage width
        style={{
          width: `${Math.min(100, usagePercentage)}%`,
        }}
      />
    </div>
  );
}
