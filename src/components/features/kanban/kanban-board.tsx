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
import {
  alignBoardColumns,
  columnsForLane,
  laneCards,
  normalizeColumnName,
  type SwimlaneNode,
} from "#/utils/kanban-swimlanes";
import { cn } from "#/utils/utils";
import { KanbanColumn } from "./kanban-column";
import { KanbanSwimlaneHeader } from "./kanban-swimlane-header";

export interface KanbanBoardViewProps {
  board: KanbanBoard;
  boards?: KanbanBoard[];
  costs?: KanbanBoardCosts | null;
  lanes?: SwimlaneNode[];
  collapsedLaneIds?: string[];
  onToggleLane?: (laneId: string) => void;
  onSelectCard?: (card: KanbanCard) => void;
  onAddCard?: (columnId: string, title: string, laneId: string | null) => void;
  onMoveCard?: (
    cardId: string,
    columnId: string,
    position: number,
    laneId: string | null,
  ) => void;
  onAddColumn?: (name: string) => void;
}

export function KanbanBoardView({
  board,
  boards,
  costs,
  lanes = [],
  collapsedLaneIds = [],
  onToggleLane,
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
  const allBoards = boards && boards.length > 0 ? boards : [board];
  const aligned = alignBoardColumns(allBoards);
  const boardById = new Map(allBoards.map((item) => [item.id, item]));

  const renderColumns = (
    sourceBoard: KanbanBoard,
    cards: KanbanCard[],
    laneId: string | null,
    hideHeader: boolean,
  ) =>
    columnsForLane(sourceBoard, aligned, cards).map((column) => {
      const realColumn = sourceBoard.columns.some(
        (item) => item.id === column.id,
      );
      return (
        <KanbanColumn
          key={`${laneId ?? "board"}-${column.id}`}
          column={column}
          aggregateCost={costByColumn.get(column.id) ?? 0}
          hideHeader={hideHeader}
          laneId={laneId}
          allowDrop={realColumn}
          onSelectCard={onSelectCard}
          onAddCard={realColumn ? onAddCard : undefined}
          onDropCard={realColumn ? onMoveCard : undefined}
        />
      );
    });

  const renderLane = (lane: SwimlaneNode) => {
    const collapsed = collapsedLaneIds.includes(lane.id);
    const sourceBoard = boardById.get(lane.boardId) ?? board;
    const hasChildren = lane.children.length > 0;
    return (
      <div key={lane.id}>
        <KanbanSwimlaneHeader
          lane={lane}
          collapsed={collapsed}
          onToggle={(id) => onToggleLane?.(id)}
        />
        {collapsed ? null : hasChildren ? (
          lane.children.map(renderLane)
        ) : (
          <div className="flex min-h-[8rem] gap-4">
            {renderColumns(sourceBoard, laneCards(lane), lane.id, true)}
          </div>
        )}
      </div>
    );
  };

  if (lanes.length > 0) {
    return (
      <div
        data-testid="kanban-board"
        className="flex h-full min-h-0 flex-1 flex-col overflow-auto"
      >
        <div className="sticky top-0 z-10 flex min-w-max gap-4 bg-[var(--oh-background)] pb-2">
          {aligned.map((column) => {
            const sample = allBoards
              .flatMap((boardItem) => boardItem.columns)
              .find((item) => normalizeColumnName(item.name) === column.key);
            return (
              <div
                key={column.key}
                className="flex min-w-[17rem] flex-1 items-center gap-2 px-1 pt-0.5"
              >
                {sample?.color ? (
                  <span
                    className="h-2 w-2 shrink-0 rounded-full"
                    style={{ backgroundColor: sample.color }}
                    aria-hidden
                  />
                ) : null}
                <h2 className="truncate text-[13px] font-medium leading-5 text-[var(--oh-foreground)]">
                  {column.name}
                </h2>
              </div>
            );
          })}
        </div>
        <div className="min-w-max flex-1">{lanes.map(renderLane)}</div>
      </div>
    );
  }

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
