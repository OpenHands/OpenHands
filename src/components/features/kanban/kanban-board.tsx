import { Plus } from "lucide-react";
import React from "react";
import { useTranslation } from "react-i18next";
import type {
  KanbanBoard,
  KanbanBoardCosts,
  KanbanCard,
} from "#/api/kanban-service/kanban-types";
import { I18nKey } from "#/i18n/declaration";
import { formControlFieldClassName } from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";
import { KanbanColumn } from "./kanban-column";

export interface KanbanBoardViewProps {
  board: KanbanBoard;
  costs?: KanbanBoardCosts | null;
  onSelectCard?: (card: KanbanCard) => void;
  onAddCard?: (columnId: string, title: string) => void;
  onMoveCard?: (cardId: string, columnId: string, position: number) => void;
  onAddColumn?: (name: string) => void;
}

export function KanbanBoardView({
  board,
  costs,
  onSelectCard,
  onAddCard,
  onMoveCard,
  onAddColumn,
}: KanbanBoardViewProps) {
  const { t } = useTranslation("openhands");
  const [addingColumn, setAddingColumn] = React.useState(false);
  const costByColumn = new Map(
    (costs?.columns ?? []).map((column) => [
      column.id,
      column.actual_cost || column.estimate_cost,
    ]),
  );

  return (
    <div
      data-testid="kanban-board"
      className="flex h-full min-h-0 flex-1 gap-4 overflow-x-auto pb-1"
    >
      {board.columns.map((column) => (
        <KanbanColumn
          key={column.id}
          column={column}
          aggregateCost={costByColumn.get(column.id) ?? 0}
          onSelectCard={onSelectCard}
          onAddCard={onAddCard}
          onDropCard={onMoveCard}
        />
      ))}
      {onAddColumn ? (
        addingColumn ? (
          <form
            className="flex h-fit w-[17rem] shrink-0 flex-col"
            onSubmit={(event) => {
              event.preventDefault();
              const form = event.currentTarget;
              const input = form.elements.namedItem(
                "columnName",
              ) as HTMLInputElement | null;
              const name = input?.value.trim() ?? "";
              if (!name) return;
              onAddColumn(name);
              form.reset();
              setAddingColumn(false);
            }}
          >
            <input
              name="columnName"
              data-testid="kanban-column-name"
              // Composer is opened by an explicit click; keep the caret in the field.
              // eslint-disable-next-line jsx-a11y/no-autofocus
              autoFocus
              placeholder={t(I18nKey.KANBAN$NEW_COLUMN_NAME)}
              aria-label={t(I18nKey.KANBAN$ADD_COLUMN)}
              onBlur={(event) => {
                if (!event.currentTarget.value.trim()) setAddingColumn(false);
              }}
              className={cn(formControlFieldClassName, "bg-transparent")}
            />
            <button
              type="submit"
              data-testid="kanban-add-column"
              className="sr-only"
            >
              {t(I18nKey.KANBAN$ADD_COLUMN)}
            </button>
          </form>
        ) : (
          <button
            type="button"
            data-testid="kanban-add-column"
            onClick={() => setAddingColumn(true)}
            aria-label={t(I18nKey.KANBAN$ADD_COLUMN)}
            className={cn(
              "flex h-8 w-8 shrink-0 items-center justify-center rounded-md",
              "text-[var(--oh-muted)] hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--oh-foreground)]",
              "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/20",
            )}
          >
            <Plus className="h-4 w-4" aria-hidden />
          </button>
        )
      ) : null}
    </div>
  );
}
