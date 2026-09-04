import { useTranslation } from "react-i18next";
import type { KanbanBoardCosts } from "#/api/kanban-service/kanban-types";
import { I18nKey } from "#/i18n/declaration";
import { formatUsd } from "./kanban-cost";

export interface CostSummaryProps {
  costs: KanbanBoardCosts;
}

export function CostSummary({ costs }: CostSummaryProps) {
  const { t } = useTranslation("openhands");
  const total = costs.total_actual_cost || costs.total_estimate_cost;

  return (
    <div
      data-testid="kanban-cost-summary"
      className="flex items-baseline gap-2 text-sm leading-5 text-tertiary-light"
    >
      <span>{t(I18nKey.KANBAN$TOTAL_COST)}</span>
      <span
        data-testid="kanban-board-total-cost"
        className="tabular-nums font-medium text-white"
      >
        {formatUsd(total)}
      </span>
    </div>
  );
}
