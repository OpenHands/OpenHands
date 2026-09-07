import { useTranslation } from "react-i18next";
import type { KanbanBoardCosts } from "#/api/kanban-service/kanban-types";
import { I18nKey } from "#/i18n/declaration";
import { CostText } from "#/components/shared/cost-text";

export interface CostSummaryProps {
  costs: KanbanBoardCosts;
}

export function CostSummary({ costs }: CostSummaryProps) {
  const { t } = useTranslation("openhands");
  const total = costs.total_actual_cost || costs.total_estimate_cost;

  // Board totals arrive pre-summed in USD without per-card dates, so convert
  // at today's rate. Column headers re-sum converted cards when cards exist.

  return (
    <div
      data-testid="kanban-cost-summary"
      className="flex items-baseline gap-1.5 text-xs leading-4 text-[var(--oh-muted)]"
    >
      <span>{t(I18nKey.KANBAN$TOTAL_COST)}</span>
      <span
        data-testid="kanban-board-total-cost"
        className="tabular-nums text-[var(--oh-foreground)]"
      >
        <CostText amount={total} />
      </span>
    </div>
  );
}
