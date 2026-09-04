import React from "react";
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
  const cards = column.cards ?? [];

  const handleDrop = (
    event: React.DragEvent<HTMLElement>,
    position: number,
  ) => {
    event.preventDefault();
    const cardId = event.dataTransfer.getData("text/plain");
    if (cardId) onDropCard?.(cardId, column.id, position);
  };

  return (
    <section
      data-testid={`kanban-column-${column.id}`}
      className={cn(
        "flex h-full w-72 shrink-0 flex-col overflow-hidden rounded-xl bg-base-secondary",
        column.color && "border-t-2",
      )}
      style={column.color ? { borderTopColor: column.color } : undefined}
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => handleDrop(event, cards.length)}
    >
      <header className="flex items-center justify-between gap-2 px-3 pb-2 pt-3">
        <div className="flex min-w-0 items-center gap-2">
          <h2 className="truncate text-sm font-medium leading-5 text-white">
            {column.name}
          </h2>
          <span className="text-xs tabular-nums text-tertiary-light">
            {cards.length}
          </span>
        </div>
        <span
          data-testid={`kanban-column-cost-${column.id}`}
          className="shrink-0 text-xs tabular-nums text-tertiary-light"
        >
          {formatUsd(aggregateCost)}
        </span>
      </header>
      <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-3">
        {cards.length === 0 ? (
          <p className="px-0.5 py-2 text-xs leading-4 text-tertiary-light">
            {t(I18nKey.KANBAN$EMPTY_COLUMN)}
          </p>
        ) : (
          cards.map((card, index) => (
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
          ))
        )}
      </div>
      <form
        className="p-3"
        onSubmit={(event) => {
          event.preventDefault();
          const title = draft.trim();
          if (!title) return;
          onAddCard?.(column.id, title);
          setDraft("");
        }}
      >
        <input
          data-testid={`kanban-add-card-input-${column.id}`}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder={t(I18nKey.KANBAN$NEW_CARD_TITLE)}
          aria-label={t(I18nKey.KANBAN$ADD_CARD)}
          className={cn(formControlFieldClassName, "bg-transparent")}
        />
        <button
          type="submit"
          data-testid={`kanban-add-card-${column.id}`}
          className="sr-only"
        >
          {t(I18nKey.KANBAN$ADD_CARD)}
        </button>
      </form>
    </section>
  );
}
