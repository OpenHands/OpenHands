import React from "react";
import { useTranslation } from "react-i18next";
import { Columns3, List } from "lucide-react";
import {
  KANBAN_VIEW_BOARD,
  KANBAN_VIEW_LIST,
} from "#/api/kanban-service/kanban-constants";
import type { KanbanCard } from "#/api/kanban-service/kanban-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { CardDetailPanel } from "#/components/features/kanban/card-detail-panel";
import { CostSummary } from "#/components/features/kanban/cost-summary";
import { KanbanBoardView } from "#/components/features/kanban/kanban-board";
import { KanbanList } from "#/components/features/kanban/kanban-list";
import { SegmentedToggle } from "#/components/features/files-tab/segmented-toggle";
import { I18nKey } from "#/i18n/declaration";
import {
  useCreateKanbanBoard,
  useCreateKanbanCard,
  useCreateKanbanColumn,
  useDeleteKanbanCard,
  useKanbanBoard,
  useKanbanBoardCosts,
  useKanbanBoards,
  useMoveKanbanCard,
  useUpdateKanbanCard,
} from "#/hooks/query/use-kanban";
import { settingsLikeMainScrollClassName } from "#/utils/settings-like-page-layout-classes";

type KanbanView = typeof KANBAN_VIEW_BOARD | typeof KANBAN_VIEW_LIST;

export default function KanbanPage() {
  const { t } = useTranslation("openhands");
  const [view, setView] = React.useState<KanbanView>(KANBAN_VIEW_BOARD);
  const [selectedCard, setSelectedCard] = React.useState<KanbanCard | null>(
    null,
  );
  const [boardName, setBoardName] = React.useState("");

  const boardsQuery = useKanbanBoards();
  const selectedBoardId = boardsQuery.data?.[0]?.id ?? null;
  const boardQuery = useKanbanBoard(selectedBoardId);
  const costsQuery = useKanbanBoardCosts(selectedBoardId);
  const createBoard = useCreateKanbanBoard();
  const createCard = useCreateKanbanCard(selectedBoardId ?? "");
  const createColumn = useCreateKanbanColumn(selectedBoardId ?? "");
  const moveCard = useMoveKanbanCard(selectedBoardId ?? "");
  const updateCard = useUpdateKanbanCard(selectedBoardId ?? "");
  const deleteCard = useDeleteKanbanCard(selectedBoardId ?? "");

  const board = boardQuery.data;
  const selectedFromBoard = board
    ? (board.columns
        .flatMap((column) => column.cards ?? [])
        .find((card) => card.id === selectedCard?.id) ?? null)
    : null;

  return (
    <main
      data-testid="kanban-page"
      aria-label={t(I18nKey.KANBAN$NAV)}
      className={settingsLikeMainScrollClassName}
    >
      <header className="mb-3 flex h-9 shrink-0 items-center justify-between gap-3">
        <span className="text-[13px] font-medium text-[var(--oh-foreground)]">
          {t(I18nKey.KANBAN$NAV)}
        </span>
        <div className="flex shrink-0 items-center gap-2">
          {costsQuery.data ? <CostSummary costs={costsQuery.data} /> : null}
          {board ? (
            <SegmentedToggle
              value={view}
              onChange={setView}
              ariaLabel={t(I18nKey.KANBAN$VIEW_MODE)}
              testId="kanban-view"
              options={[
                {
                  value: KANBAN_VIEW_BOARD,
                  label: t(I18nKey.KANBAN$BOARD_VIEW),
                  icon: <Columns3 className="h-3.5 w-3.5" aria-hidden />,
                },
                {
                  value: KANBAN_VIEW_LIST,
                  label: t(I18nKey.KANBAN$LIST_VIEW),
                  icon: <List className="h-3.5 w-3.5" aria-hidden />,
                },
              ]}
            />
          ) : null}
        </div>
      </header>

      {!selectedBoardId ? (
        <form
          className="mt-6 flex max-w-md flex-col gap-3"
          onSubmit={(event) => {
            event.preventDefault();
            const name = boardName.trim();
            if (!name) return;
            createBoard.mutate({ name });
            setBoardName("");
          }}
        >
          <p data-testid="kanban-empty">{t(I18nKey.KANBAN$NO_BOARDS)}</p>
          <input
            data-testid="kanban-board-name"
            value={boardName}
            onChange={(event) => setBoardName(event.target.value)}
            placeholder={t(I18nKey.KANBAN$BOARD_NAME)}
            className="min-w-0 flex-1 rounded-md border border-[var(--oh-border)] bg-transparent px-3 py-2"
          />
          <BrandButton
            type="submit"
            variant="primary"
            testId="kanban-create-board"
          >
            {t(I18nKey.KANBAN$CREATE_BOARD)}
          </BrandButton>
        </form>
      ) : null}

      {board && view === KANBAN_VIEW_BOARD ? (
        <div className="mt-4 flex min-h-0 flex-1">
          <KanbanBoardView
            board={board}
            costs={costsQuery.data}
            onSelectCard={setSelectedCard}
            onAddCard={(columnId, title) =>
              createCard.mutate({ columnId, payload: { title } })
            }
            onMoveCard={(cardId, columnId, position) =>
              moveCard.mutate({
                cardId,
                payload: { column_id: columnId, position },
              })
            }
            onAddColumn={(name) => createColumn.mutate({ name })}
          />
        </div>
      ) : null}

      {board && view === KANBAN_VIEW_LIST ? (
        <div className="mt-4">
          <KanbanList board={board} onSelectCard={setSelectedCard} />
        </div>
      ) : null}

      {selectedFromBoard ? (
        <div className="fixed inset-y-0 right-0 z-20">
          <CardDetailPanel
            card={selectedFromBoard}
            onClose={() => setSelectedCard(null)}
            onUpdate={(cardId, payload) =>
              updateCard.mutate({ cardId, payload })
            }
            onDelete={(cardId) => {
              deleteCard.mutate(cardId);
              setSelectedCard(null);
            }}
          />
        </div>
      ) : null}
    </main>
  );
}
