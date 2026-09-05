import { GitBranch, GitPullRequest } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { KanbanCard } from "#/api/kanban-service/kanban-types";
import { KANBAN_DONE_STATUS } from "#/api/kanban-service/kanban-constants";
import { I18nKey } from "#/i18n/declaration";
import { formatRelativeTime } from "#/utils/format-relative-time";
import { formControlTransitionClassName } from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";
import { cardDisplayCost, formatUsd } from "./kanban-cost";
import { pullRequestChipLabel } from "./kanban-pr-label";

const PRIORITY_CLASS: Record<string, string> = {
  P0: "text-red-400",
  P1: "text-orange-300",
  P2: "text-[var(--oh-muted)]",
  P3: "text-[var(--oh-muted)]",
};

const META_CHIP_CLASS =
  "inline-flex max-w-full min-w-0 items-center gap-1 truncate rounded-md bg-white/[0.04] px-1.5 py-0.5 text-[11px] leading-4 text-[var(--oh-muted)]";

export interface KanbanCardProps {
  card: KanbanCard;
  onSelect?: (card: KanbanCard) => void;
}

export function KanbanCard({ card, onSelect }: KanbanCardProps) {
  const { t, i18n } = useTranslation("openhands");
  const cost = cardDisplayCost(card);
  const costLabel =
    cost.kind === "actual"
      ? t(I18nKey.KANBAN$COST_ACTUAL)
      : t(I18nKey.KANBAN$COST_ESTIMATE);
  const isLive =
    Boolean(card.agent_session_id) && card.status !== KANBAN_DONE_STATUS;
  const updatedLabel = card.updated_at
    ? formatRelativeTime(card.updated_at, i18n.language, t)
    : null;

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
        "w-full rounded-lg border border-[var(--oh-border-subtle)] bg-[var(--oh-surface-raised)]/50 p-2.5 text-left",
        formControlTransitionClassName,
        "hover:border-[var(--oh-border)] hover:bg-[var(--oh-interactive-hover)]",
        "focus-visible:border-white/40 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/20",
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="min-w-0 text-[13px] font-medium leading-5 text-[var(--oh-foreground)]">
          {card.title}
        </span>
        <span
          data-testid={`kanban-card-priority-${card.id}`}
          className={cn(
            "shrink-0 text-[11px] font-medium leading-4",
            PRIORITY_CLASS[card.priority] ?? PRIORITY_CLASS.P2,
          )}
        >
          {card.priority}
        </span>
      </div>
      {isLive || card.linked_pr || card.linked_branch ? (
        <div className="mt-2 flex min-w-0 flex-wrap items-center gap-1.5">
          {isLive ? (
            <span
              data-testid={`kanban-card-live-${card.id}`}
              className={cn(META_CHIP_CLASS, "text-emerald-300")}
            >
              <span
                className="h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-400"
                aria-hidden
              />
              {t(I18nKey.KANBAN$LIVE)}
            </span>
          ) : null}
          {card.linked_pr ? (
            <span
              data-testid={`kanban-card-pr-${card.id}`}
              className={META_CHIP_CLASS}
            >
              <GitPullRequest className="h-3 w-3 shrink-0" aria-hidden />
              {pullRequestChipLabel(card.linked_pr)}
            </span>
          ) : null}
          {card.linked_branch ? (
            <span className={META_CHIP_CLASS}>
              <GitBranch className="h-3 w-3 shrink-0" aria-hidden />
              <span className="truncate">{card.linked_branch}</span>
            </span>
          ) : null}
        </div>
      ) : null}
      <div className="mt-2 flex items-center justify-between gap-2 text-[11px] leading-4 text-[var(--oh-muted)]">
        <span className="min-w-0 truncate">
          {[card.assignee, updatedLabel].filter(Boolean).join(" · ")}
        </span>
        <span className="flex shrink-0 items-center gap-1 tabular-nums">
          <span data-testid={`kanban-card-cost-kind-${card.id}`}>
            {costLabel}
          </span>
          <span
            data-testid={`kanban-card-cost-${card.id}`}
            className="text-[var(--oh-foreground)]"
          >
            {formatUsd(cost.amount)}
          </span>
        </span>
      </div>
    </button>
  );
}
