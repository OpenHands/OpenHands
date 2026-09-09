import { KANBAN_DONE_STATUS } from "#/api/kanban-service/kanban-constants";
import type { KanbanCard } from "#/api/kanban-service/kanban-types";

export { formatUsd } from "#/stores/cost-currency-store";

export function cardDisplayCost(card: KanbanCard): {
  amount: number;
  kind: "actual" | "estimate";
} {
  const hasActuals =
    card.status === KANBAN_DONE_STATUS ||
    (card.actual_cost != null && card.actual_cost > 0);
  if (hasActuals) {
    return { amount: Number(card.actual_cost ?? 0), kind: "actual" };
  }
  return { amount: Number(card.estimate_cost ?? 0), kind: "estimate" };
}

export function cardCostAt(card: KanbanCard): string {
  return cardDisplayCost(card).kind === "actual"
    ? card.updated_at
    : card.created_at;
}
