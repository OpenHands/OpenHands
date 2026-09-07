import React from "react";
import { useTranslation } from "react-i18next";
import type {
  KanbanBoard,
  KanbanCard,
} from "#/api/kanban-service/kanban-types";
import { I18nKey } from "#/i18n/declaration";
import { formatDate } from "#/utils/format-relative-time";
import { cn } from "#/utils/utils";
import { CostText } from "#/components/shared/cost-text";
import type { SwimlaneNode } from "#/utils/kanban-swimlanes";
import { laneCards } from "#/utils/kanban-swimlanes";
import { cardCostAt, cardDisplayCost } from "./kanban-cost";

export type KanbanListSortKey =
  | "priority"
  | "status"
  | "assignee"
  | "cost"
  | "created";

const PRIORITY_RANK: Record<string, number> = {
  P0: 0,
  P1: 1,
  P2: 2,
  P3: 3,
};

function flattenCards(board: KanbanBoard): KanbanCard[] {
  return board.columns.flatMap((column) => column.cards ?? []);
}

function compareCards(
  left: KanbanCard,
  right: KanbanCard,
  sortKey: KanbanListSortKey,
): number {
  switch (sortKey) {
    case "priority":
      return (
        (PRIORITY_RANK[left.priority] ?? 9) -
        (PRIORITY_RANK[right.priority] ?? 9)
      );
    case "status":
      return left.status.localeCompare(right.status);
    case "assignee":
      return (left.assignee ?? "").localeCompare(right.assignee ?? "");
    case "cost":
      return cardDisplayCost(left).amount - cardDisplayCost(right).amount;
    case "created":
      return left.created_at.localeCompare(right.created_at);
    default:
      return 0;
  }
}

function sortCards(
  cards: KanbanCard[],
  sortKey: KanbanListSortKey,
): KanbanCard[] {
  return cards
    .slice()
    .sort((left, right) => compareCards(left, right, sortKey));
}

export interface KanbanListProps {
  board: KanbanBoard;
  lanes?: SwimlaneNode[];
  onSelectCard?: (card: KanbanCard) => void;
}

export function KanbanList({
  board,
  lanes = [],
  onSelectCard,
}: KanbanListProps) {
  const { t, i18n } = useTranslation("openhands");
  const [sortKey, setSortKey] = React.useState<KanbanListSortKey>("priority");
  const rows = sortCards(flattenCards(board), sortKey);

  const header = (key: KanbanListSortKey, label: string) => (
    <th className="px-3 py-2 font-medium">
      <button
        type="button"
        data-testid={`kanban-list-sort-${key}`}
        className={cn(
          "text-left text-xs leading-4 text-tertiary-light",
          sortKey === key && "text-white",
        )}
        onClick={() => setSortKey(key)}
      >
        {label}
      </button>
    </th>
  );

  const renderCardRow = (card: KanbanCard) => (
    <tr
      key={card.id}
      data-testid={`kanban-list-row-${card.id}`}
      className="cursor-pointer border-b border-[var(--oh-border)] last:border-b-0 hover:bg-[var(--oh-interactive-hover)]"
      onClick={() => onSelectCard?.(card)}
    >
      <td className="px-3 py-2.5 font-medium text-white">{card.title}</td>
      <td className="px-3 py-2.5 text-tertiary-light">{card.priority}</td>
      <td className="px-3 py-2.5 text-tertiary-light">{card.status}</td>
      <td className="px-3 py-2.5 text-tertiary-light">{card.assignee ?? ""}</td>
      <td className="px-3 py-2.5 tabular-nums text-white">
        <CostText amount={cardDisplayCost(card).amount} at={cardCostAt(card)} />
      </td>
      <td className="px-3 py-2.5 tabular-nums text-tertiary-light">
        {formatDate(card.created_at, i18n.language)}
      </td>
    </tr>
  );

  const renderLaneRows = (lane: SwimlaneNode): React.ReactNode[] => {
    const cards = sortCards(laneCards(lane), sortKey);
    const label =
      lane.kind === "ungrouped" ? t(I18nKey.KANBAN$UNGROUPED) : lane.name;
    const usd = cards.reduce(
      (sum, card) => sum + cardDisplayCost(card).amount,
      0,
    );
    const headerRow = (
      <tr
        key={`lane-${lane.id}`}
        data-testid={`kanban-list-lane-${lane.id}`}
        className="border-b border-[var(--oh-border)] bg-[var(--oh-surface-raised)]/40"
      >
        <td
          colSpan={6}
          className={cn(
            "px-3 py-2 text-xs font-medium text-[var(--oh-foreground)]",
            lane.depth === 1 && "pl-8",
          )}
        >
          <span className="inline-flex items-center gap-2">
            <span>{label}</span>
            <span className="text-[var(--oh-muted)]">{cards.length}</span>
            <span
              data-testid={`kanban-list-lane-cost-${lane.id}`}
              className="tabular-nums text-[var(--oh-muted)]"
            >
              <CostText
                amount={usd}
                at={cards[0] ? cardCostAt(cards[0]) : undefined}
              />
            </span>
          </span>
        </td>
      </tr>
    );
    if (lane.children.length > 0) {
      return [headerRow, ...lane.children.flatMap(renderLaneRows)];
    }
    return [headerRow, ...cards.map(renderCardRow)];
  };

  return (
    <div data-testid="kanban-list" className="h-full min-h-0 overflow-auto">
      <table className="w-full min-w-[640px] text-left text-sm">
        <thead className="sticky top-0 bg-base">
          <tr className="border-b border-[var(--oh-border)]">
            <th className="px-3 py-2 text-xs font-medium text-tertiary-light">
              {t(I18nKey.KANBAN$TITLE)}
            </th>
            {header("priority", t(I18nKey.KANBAN$PRIORITY))}
            {header("status", t(I18nKey.COMMON$STATUS))}
            {header("assignee", t(I18nKey.KANBAN$ASSIGNEE))}
            {header("cost", t(I18nKey.KANBAN$COST))}
            {header("created", t(I18nKey.KANBAN$CREATED))}
          </tr>
        </thead>
        <tbody>
          {lanes.length > 0
            ? lanes.flatMap(renderLaneRows)
            : rows.map(renderCardRow)}
        </tbody>
      </table>
    </div>
  );
}
