import React from "react";
import { useTranslation } from "react-i18next";
import type {
  KanbanBoard,
  KanbanCard,
} from "#/api/kanban-service/kanban-types";
import { I18nKey } from "#/i18n/declaration";
import { formatDate } from "#/utils/format-relative-time";
import { cn } from "#/utils/utils";
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
  const { t, i18n } = useTranslation("openhands");
  const [sortKey, setSortKey] = React.useState<KanbanListSortKey>("priority");
  const rows = flattenCards(board)
    .slice()
    .sort((a, b) => compareCards(a, b, sortKey));

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

  return (
    <div
      data-testid="kanban-list"
      className="h-full min-h-0 overflow-auto rounded-xl bg-base-secondary"
    >
      <table className="w-full min-w-[640px] text-left text-sm">
        <thead className="sticky top-0 bg-base-secondary">
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
          {rows.map((card) => (
            <tr
              key={card.id}
              data-testid={`kanban-list-row-${card.id}`}
              className="cursor-pointer border-b border-[var(--oh-border)] last:border-b-0 hover:bg-[var(--oh-interactive-hover)]"
              onClick={() => onSelectCard?.(card)}
            >
              <td className="px-3 py-2.5 font-medium text-white">
                {card.title}
              </td>
              <td className="px-3 py-2.5 text-tertiary-light">
                {card.priority}
              </td>
              <td className="px-3 py-2.5 text-tertiary-light">{card.status}</td>
              <td className="px-3 py-2.5 text-tertiary-light">
                {card.assignee ?? ""}
              </td>
              <td className="px-3 py-2.5 tabular-nums text-white">
                {formatUsd(cardDisplayCost(card).amount)}
              </td>
              <td className="px-3 py-2.5 tabular-nums text-tertiary-light">
                {formatDate(card.created_at, i18n.language)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
