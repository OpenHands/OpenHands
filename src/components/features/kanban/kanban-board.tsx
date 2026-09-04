import type {
  KanbanBoard,
  KanbanBoardCosts,
  KanbanCard,
} from "#/api/kanban-service/kanban-types";
import { KanbanColumn } from "./kanban-column";

export interface KanbanBoardViewProps {
  board: KanbanBoard;
  costs?: KanbanBoardCosts | null;
  onSelectCard?: (card: KanbanCard) => void;
  onAddCard?: (columnId: string, title: string) => void;
  onMoveCard?: (cardId: string, columnId: string, position: number) => void;
}

export function KanbanBoardView({
  board,
  costs,
  onSelectCard,
  onAddCard,
  onMoveCard,
}: KanbanBoardViewProps) {
  const costByColumn = new Map(
    (costs?.columns ?? []).map((column) => [
      column.id,
      column.actual_cost || column.estimate_cost,
    ]),
  );

  return (
    <div
      data-testid="kanban-board"
      className="flex min-h-0 flex-1 gap-3 overflow-x-auto pb-4"
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
    </div>
  );
}
