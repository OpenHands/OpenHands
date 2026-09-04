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
  const costByColumn = new Map(
    (costs?.columns ?? []).map((column) => [
      column.id,
      column.actual_cost || column.estimate_cost,
    ]),
  );

  return (
    <div
      data-testid="kanban-board"
      className="flex h-full min-h-0 flex-1 gap-3 overflow-x-auto pb-1"
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
        <form
          className="flex h-full w-72 shrink-0 flex-col rounded-xl border border-dashed border-[var(--oh-border)] p-3"
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
          }}
        >
          <input
            name="columnName"
            data-testid="kanban-column-name"
            placeholder={t(I18nKey.KANBAN$NEW_COLUMN_NAME)}
            aria-label={t(I18nKey.KANBAN$ADD_COLUMN)}
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
      ) : null}
    </div>
  );
}
