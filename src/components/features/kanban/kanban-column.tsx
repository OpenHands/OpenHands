import React from "react";
import { Plus } from "lucide-react";
import { useTranslation } from "react-i18next";
import type {
  KanbanCard,
  KanbanColumn,
} from "#/api/kanban-service/kanban-types";
import { I18nKey } from "#/i18n/declaration";
import { formControlFieldClassName } from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";
import { formatUsd } from "./kanban-cost";
import { KanbanCard as KanbanCardView } from "./kanban-card";

export interface KanbanColumnProps {
  column: KanbanColumn;
  aggregateCost: number;
  onSelectCard?: (card: KanbanCard) => void;
  onAddCard?: (columnId: string, title: string) => void;
  onDropCard?: (cardId: string, columnId: string, position: number) => void;
}

export function KanbanColumn({
  column,
  aggregateCost,
  onSelectCard,
  onAddCard,
  onDropCard,
}: KanbanColumnProps) {
  const { t } = useTranslation("openhands");
  const [draft, setDraft] = React.useState("");
  const [composing, setComposing] = React.useState(false);
  const cards = column.cards ?? [];

  const handleDrop = (
    event: React.DragEvent<HTMLElement>,
    position: number,
  ) => {
    event.preventDefault();
    const cardId = event.dataTransfer.getData("text/plain");
    if (cardId) onDropCard?.(cardId, column.id, position);
  };

  const finishCompose = () => {
    const title = draft.trim();
    if (title) onAddCard?.(column.id, title);
    setDraft("");
    setComposing(false);
  };

  return (
    <section
      data-testid={`kanban-column-${column.id}`}
      className="flex h-full min-h-0 min-w-[17rem] flex-1 flex-col"
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => handleDrop(event, cards.length)}
    >
      <header className="flex items-center justify-between gap-2 px-1 pb-2 pt-0.5">
        <div className="flex min-w-0 items-center gap-2">
          {column.color ? (
            <span
              className="h-2 w-2 shrink-0 rounded-full"
              style={{ backgroundColor: column.color }}
              aria-hidden
            />
          ) : null}
          <h2 className="truncate text-[13px] font-medium leading-5 text-[var(--oh-foreground)]">
            {column.name}
          </h2>
          <span className="rounded-md bg-white/[0.06] px-1.5 py-0.5 text-[11px] tabular-nums leading-4 text-[var(--oh-muted)]">
            {cards.length}
          </span>
        </div>
        <span
          data-testid={`kanban-column-cost-${column.id}`}
          className="shrink-0 text-[11px] tabular-nums text-[var(--oh-muted)]"
        >
          {formatUsd(aggregateCost)}
        </span>
      </header>
      <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-0.5 pb-1">
        {cards.map((card, index) => (
          <div
            key={card.id}
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.stopPropagation();
              handleDrop(event, index);
            }}
          >
            <KanbanCardView card={card} onSelect={onSelectCard} />
          </div>
        ))}
        {composing ? (
          <form
            className="shrink-0"
            onSubmit={(event) => {
              event.preventDefault();
              finishCompose();
            }}
          >
            <input
              data-testid={`kanban-add-card-input-${column.id}`}
              value={draft}
              // Composer is opened by an explicit click; keep the caret in the field.
              // eslint-disable-next-line jsx-a11y/no-autofocus
              autoFocus
              onChange={(event) => setDraft(event.target.value)}
              onBlur={() => {
                if (!draft.trim()) setComposing(false);
              }}
              onKeyDown={(event) => {
                if (event.key === "Escape") {
                  setDraft("");
                  setComposing(false);
                }
              }}
              placeholder={t(I18nKey.KANBAN$NEW_CARD_TITLE)}
              aria-label={t(I18nKey.KANBAN$ADD_CARD)}
              className={cn(formControlFieldClassName, "bg-transparent")}
            />
            <button type="submit" className="sr-only">
              {t(I18nKey.KANBAN$ADD_CARD)}
            </button>
          </form>
        ) : (
          <button
            type="button"
            data-testid={`kanban-add-card-${column.id}`}
            onClick={() => setComposing(true)}
            className={cn(
              "inline-flex h-8 shrink-0 items-center gap-1.5 rounded-md px-1.5 text-[13px]",
              "text-[var(--oh-muted)] hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--oh-foreground)]",
              "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/20",
            )}
          >
            <Plus className="h-3.5 w-3.5" aria-hidden />
            {t(I18nKey.KANBAN$NEW)}
          </button>
        )}
      </div>
    </section>
  );
}
