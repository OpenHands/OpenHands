import { ChevronDown, ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  useConvertUsd,
  useFormatCostAmount,
} from "#/stores/cost-currency-store";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import type { SwimlaneNode } from "#/utils/kanban-swimlanes";
import { laneCards } from "#/utils/kanban-swimlanes";
import { cardCostAt, cardDisplayCost } from "./kanban-cost";

export interface KanbanSwimlaneHeaderProps {
  lane: SwimlaneNode;
  collapsed: boolean;
  onToggle: (laneId: string) => void;
}

export function KanbanSwimlaneHeader({
  lane,
  collapsed,
  onToggle,
}: KanbanSwimlaneHeaderProps) {
  const { t } = useTranslation("openhands");
  const convertUsd = useConvertUsd();
  const formatAmount = useFormatCostAmount();
  const cards = laneCards(lane);
  const convertedTotal = cards.reduce((sum, card) => {
    const { amount } = cardDisplayCost(card);
    return sum + convertUsd(amount, cardCostAt(card));
  }, 0);
  const label =
    lane.kind === "ungrouped" ? t(I18nKey.KANBAN$UNGROUPED) : lane.name;
  const toggleLabel = collapsed
    ? t(I18nKey.KANBAN$EXPAND_SWIMLANE)
    : t(I18nKey.KANBAN$COLLAPSE_SWIMLANE);

  return (
    <div
      data-testid={`kanban-swimlane-${lane.id}`}
      className={cn(
        "flex items-center justify-between gap-2 px-1 py-1.5",
        lane.depth === 0
          ? "border-t border-[var(--oh-border-subtle)] first:border-t-0"
          : "pl-4",
      )}
    >
      <button
        type="button"
        data-testid={`kanban-swimlane-toggle-${lane.id}`}
        aria-expanded={!collapsed}
        aria-label={toggleLabel}
        onClick={() => onToggle(lane.id)}
        className="flex min-w-0 items-center gap-1.5 text-left text-[13px] font-medium leading-5 text-[var(--oh-foreground)]"
      >
        {collapsed ? (
          <ChevronRight className="h-3.5 w-3.5 shrink-0 text-[var(--oh-muted)]" />
        ) : (
          <ChevronDown className="h-3.5 w-3.5 shrink-0 text-[var(--oh-muted)]" />
        )}
        <span className="truncate">{label}</span>
        <span className="rounded-md bg-white/[0.06] px-1.5 py-0.5 text-[11px] tabular-nums leading-4 text-[var(--oh-muted)]">
          {cards.length}
        </span>
      </button>
      <span
        data-testid={`kanban-swimlane-cost-${lane.id}`}
        className="shrink-0 text-[11px] tabular-nums text-[var(--oh-muted)]"
      >
        {formatAmount(convertedTotal)}
      </span>
    </div>
  );
}
