import { useTranslation } from "react-i18next";
import type { KanbanCard } from "#/api/kanban-service/kanban-types";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { cardDisplayCost, formatUsd } from "./kanban-cost";

const PRIORITY_CLASS: Record<string, string> = {
  P0: "bg-red-500/20 text-red-300",
  P1: "bg-orange-500/20 text-orange-300",
  P2: "bg-yellow-500/20 text-yellow-200",
  P3: "bg-zinc-500/20 text-zinc-300",
};

export interface KanbanCardProps {
  card: KanbanCard;
  onSelect?: (card: KanbanCard) => void;
}

export function KanbanCard({ card, onSelect }: KanbanCardProps) {
  const { t } = useTranslation("openhands");
  const cost = cardDisplayCost(card);
  const costLabel =
    cost.kind === "actual"
      ? t(I18nKey.KANBAN$COST_ACTUAL)
      : t(I18nKey.KANBAN$COST_ESTIMATE);

  return (
    <button
      type="button"
      data-testid={`kanban-card-${card.id}`}
      draggable
      onDragStart={(event) => {
        event.dataTransfer.setData("text/plain", card.id);
      }}
      onClick={() => onSelect?.(card)}
      className="w-full rounded-lg border border-[var(--oh-border)] bg-[var(--oh-surface-raised)] p-3 text-left hover:border-[var(--oh-accent)]"
    >
      <div className="flex items-start justify-between gap-2">
        <span className="text-sm font-medium text-[var(--foreground)]">
          {card.title}
        </span>
        <span
          data-testid={`kanban-card-priority-${card.id}`}
          className={cn(
            "shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold",
            PRIORITY_CLASS[card.priority] ?? PRIORITY_CLASS.P2,
          )}
        >
          {card.priority}
        </span>
      </div>
      <div className="mt-2 flex items-center justify-between text-xs text-[var(--oh-muted)]">
        <span data-testid={`kanban-card-cost-kind-${card.id}`}>
          {costLabel}
        </span>
        <span data-testid={`kanban-card-cost-${card.id}`}>
          {formatUsd(cost.amount)}
        </span>
      </div>
    </button>
  );
}
