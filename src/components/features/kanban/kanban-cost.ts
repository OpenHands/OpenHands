import { KANBAN_DONE_STATUS } from "#/api/kanban-service/kanban-constants";
import type { KanbanCard } from "#/api/kanban-service/kanban-types";

export function formatUsd(amount: number | null | undefined): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(Number(amount ?? 0));
}

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
