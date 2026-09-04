import React from "react";
import { useTranslation } from "react-i18next";
import type {
  KanbanBoard,
  KanbanCard,
} from "#/api/kanban-service/kanban-types";
import { I18nKey } from "#/i18n/declaration";
import { cardDisplayCost, formatUsd } from "./kanban-cost";

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

export interface KanbanListProps {
  board: KanbanBoard;
  onSelectCard?: (card: KanbanCard) => void;
}

export function KanbanList({ board, onSelectCard }: KanbanListProps) {
  const { t } = useTranslation("openhands");
  const [sortKey, setSortKey] = React.useState<KanbanListSortKey>("priority");
  const rows = flattenCards(board)
    .slice()
    .sort((a, b) => compareCards(a, b, sortKey));

  const header = (key: KanbanListSortKey, label: string) => (
    <th>
      <button
        type="button"
        data-testid={`kanban-list-sort-${key}`}
        className="font-semibold"
        onClick={() => setSortKey(key)}
      >
        {label}
      </button>
    </th>
  );

  return (
    <div data-testid="kanban-list" className="overflow-x-auto">
      <table className="w-full min-w-[640px] text-left text-sm">
        <thead>
          <tr className="border-b border-[var(--oh-border)] text-[var(--oh-muted)]">
            <th>{t(I18nKey.KANBAN$TITLE)}</th>
            {header("priority", t(I18nKey.KANBAN$PRIORITY))}
            {header("status", t(I18nKey.COMMON$STATUS))}
            {header("assignee", t(I18nKey.KANBAN$ASSIGNEE))}
            {header("cost", t(I18nKey.KANBAN$COST))}
            {header("created", t(I18nKey.KANBAN$CREATED))}
          </tr>
        </thead>
        <tbody>
          {rows.map((card) => (
            <tr
              key={card.id}
              data-testid={`kanban-list-row-${card.id}`}
              className="cursor-pointer border-b border-[var(--oh-border)] hover:bg-[var(--oh-surface-raised)]"
              onClick={() => onSelectCard?.(card)}
            >
              <td className="py-2">{card.title}</td>
              <td>{card.priority}</td>
              <td>{card.status}</td>
              <td>{card.assignee ?? ""}</td>
              <td>{formatUsd(cardDisplayCost(card).amount)}</td>
              <td>{card.created_at}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
