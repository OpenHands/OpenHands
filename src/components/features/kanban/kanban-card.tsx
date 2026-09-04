import { useTranslation } from "react-i18next";
import type { KanbanCard } from "#/api/kanban-service/kanban-types";
import { I18nKey } from "#/i18n/declaration";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import { formControlTransitionClassName } from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";
import { cardDisplayCost, formatUsd } from "./kanban-cost";

const PRIORITY_CLASS: Record<string, string> = {
  P0: "text-red-400",
  P1: "text-orange-300",
  P2: "text-tertiary-light",
  P3: "text-tertiary-light",
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
      className={cn(
        "w-full rounded-lg border border-transparent bg-[rgba(255,255,255,0.04)] p-3 text-left",
        formControlTransitionClassName,
        "hover:bg-[var(--oh-interactive-hover)] focus-visible:border-white/40 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/20",
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="min-w-0 break-words text-sm font-medium leading-5 text-white">
          {card.title}
        </span>
        <span
          data-testid={`kanban-card-priority-${card.id}`}
          className={cn(
            extensionModuleCardPillClassName,
            PRIORITY_CLASS[card.priority] ?? PRIORITY_CLASS.P2,
          )}
        >
          {card.priority}
        </span>
      </div>
      <div className="mt-2 flex items-center justify-between text-xs leading-4 text-tertiary-light">
        <span data-testid={`kanban-card-cost-kind-${card.id}`}>
          {costLabel}
        </span>
        <span
          data-testid={`kanban-card-cost-${card.id}`}
          className="tabular-nums text-white"
        >
          {formatUsd(cost.amount)}
        </span>
      </div>
    </button>
  );
}
