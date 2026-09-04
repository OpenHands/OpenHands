import React from "react";
import { useTranslation } from "react-i18next";
import {
  KANBAN_VIEW_BOARD,
  KANBAN_VIEW_LIST,
  PROJECT_INIT_PATH,
} from "#/api/kanban-service/kanban-constants";
import type { KanbanCard } from "#/api/kanban-service/kanban-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { CardDetailPanel } from "#/components/features/kanban/card-detail-panel";
import { CostSummary } from "#/components/features/kanban/cost-summary";
import { KanbanBoardView } from "#/components/features/kanban/kanban-board";
import { KanbanList } from "#/components/features/kanban/kanban-list";
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
import { useNavigation } from "#/context/navigation-context";
import { settingsLikeMainScrollClassName } from "#/utils/settings-like-page-layout-classes";

type KanbanView = typeof KANBAN_VIEW_BOARD | typeof KANBAN_VIEW_LIST;

export default function KanbanPage() {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const [view, setView] = React.useState<KanbanView>(KANBAN_VIEW_BOARD);
  const [selectedCard, setSelectedCard] = React.useState<KanbanCard | null>(
    null,
  );
  const [boardName, setBoardName] = React.useState("");
  const [columnName, setColumnName] = React.useState("");

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
    <main data-testid="kanban-page" className={settingsLikeMainScrollClassName}>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-semibold">{t(I18nKey.KANBAN$NAV)}</h1>
        <div className="flex items-center gap-2">
          <BrandButton
            type="button"
            variant="secondary"
            testId="kanban-new-project"
            onClick={() => navigate(PROJECT_INIT_PATH)}
          >
            {t(I18nKey.PROJECT_INIT$NAV)}
          </BrandButton>
          <BrandButton
            type="button"
            variant={view === KANBAN_VIEW_BOARD ? "primary" : "secondary"}
            testId="kanban-view-board"
            onClick={() => setView(KANBAN_VIEW_BOARD)}
          >
            {t(I18nKey.KANBAN$BOARD_VIEW)}
          </BrandButton>
          <BrandButton
            type="button"
            variant={view === KANBAN_VIEW_LIST ? "primary" : "secondary"}
            testId="kanban-view-list"
            onClick={() => setView(KANBAN_VIEW_LIST)}
          >
            {t(I18nKey.KANBAN$LIST_VIEW)}
          </BrandButton>
        </div>
      </div>

      {costsQuery.data ? <CostSummary costs={costsQuery.data} /> : null}

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
        <div className="mt-4 flex min-h-0 flex-1 gap-3">
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
          />
          <form
            className="flex w-56 shrink-0 flex-col gap-2"
            onSubmit={(event) => {
              event.preventDefault();
              const name = columnName.trim();
              if (!name || !selectedBoardId) return;
              createColumn.mutate({ name });
              setColumnName("");
            }}
          >
            <input
              data-testid="kanban-column-name"
              value={columnName}
              onChange={(event) => setColumnName(event.target.value)}
              placeholder={t(I18nKey.KANBAN$NEW_COLUMN_NAME)}
              className="rounded-md border border-[var(--oh-border)] bg-transparent px-2 py-1 text-sm"
            />
            <BrandButton
              type="submit"
              variant="secondary"
              testId="kanban-add-column"
            >
              {t(I18nKey.KANBAN$ADD_COLUMN)}
            </BrandButton>
          </form>
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
