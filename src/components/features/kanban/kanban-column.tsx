import React from "react";
import { useTranslation } from "react-i18next";
import type {
  KanbanCard,
  KanbanColumn,
} from "#/api/kanban-service/kanban-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
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
      className="flex w-72 shrink-0 flex-col rounded-xl border border-[var(--oh-border)] bg-[var(--oh-surface)]"
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => handleDrop(event, cards.length)}
    >
      <header
        className="flex items-center justify-between gap-2 border-b border-[var(--oh-border)] px-3 py-2"
        style={column.color ? { borderTopColor: column.color } : undefined}
      >
        <h2 className="text-sm font-semibold text-[var(--foreground)]">
          {column.name}
        </h2>
        <span
          data-testid={`kanban-column-cost-${column.id}`}
          className="text-xs text-[var(--oh-muted)]"
        >
          {formatUsd(aggregateCost)}
        </span>
      </header>
      <div className="flex flex-1 flex-col gap-2 p-2">
        {cards.length === 0 ? (
          <p className="px-1 text-xs text-[var(--oh-muted)]">
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
        className="flex gap-2 border-t border-[var(--oh-border)] p-2"
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
          className="min-w-0 flex-1 rounded-md border border-[var(--oh-border)] bg-transparent px-2 py-1 text-sm"
        />
        <BrandButton
          type="submit"
          variant="secondary"
          testId={`kanban-add-card-${column.id}`}
        >
          {t(I18nKey.KANBAN$ADD_CARD)}
        </BrandButton>
      </form>
    </section>
  );
}
