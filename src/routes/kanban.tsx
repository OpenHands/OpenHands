import React from "react";
import { useTranslation } from "react-i18next";
import { Columns3, FolderPlus, List } from "lucide-react";
import {
  KANBAN_VIEW_BOARD,
  KANBAN_VIEW_LIST,
  PROJECT_INIT_PATH,
} from "#/api/kanban-service/kanban-constants";
import type { KanbanCard } from "#/api/kanban-service/kanban-types";
import { CardDetailPanel } from "#/components/features/kanban/card-detail-panel";
import { CostSummary } from "#/components/features/kanban/cost-summary";
import { KanbanBoardView } from "#/components/features/kanban/kanban-board";
import { KanbanList } from "#/components/features/kanban/kanban-list";
import { KanbanWorkspacePicker } from "#/components/features/kanban/kanban-workspace-picker";
import { boardForWorkspace } from "#/components/features/kanban/kanban-workspace";
import { SegmentedToggle } from "#/components/features/files-tab/segmented-toggle";
import { useNavigation } from "#/context/navigation-context";
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
import { useKanbanWorkspace } from "#/hooks/use-kanban-workspace";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { I18nKey } from "#/i18n/declaration";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";
import { kanbanPageShellClassName } from "#/utils/kanban-page-layout-classes";

type KanbanView = typeof KANBAN_VIEW_BOARD | typeof KANBAN_VIEW_LIST;

export default function KanbanPage() {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const [view, setView] = React.useState<KanbanView>(KANBAN_VIEW_BOARD);
  const [selectedCard, setSelectedCard] = React.useState<KanbanCard | null>(
    null,
  );
  const creatingPathRef = React.useRef<string | null>(null);

  const workspace = useKanbanWorkspace();
  const boardsQuery = useKanbanBoards();
  const selectedBoard = boardForWorkspace(
    boardsQuery.data ?? [],
    workspace.selected?.path ?? null,
  );
  const selectedBoardId = selectedBoard?.id ?? null;
  const boardQuery = useKanbanBoard(selectedBoardId);
  const costsQuery = useKanbanBoardCosts(selectedBoardId);
  const { mutate: createBoard, isPending: isCreateBoardPending } =
    useCreateKanbanBoard();
  const failedCreatePathsRef = React.useRef(new Set<string>());
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

  React.useEffect(() => {
    setSelectedCard(null);
  }, [workspace.selected?.path]);

  React.useEffect(() => {
    const selected = workspace.selected;
    if (!selected?.path || boardsQuery.isLoading) return;
    if (selectedBoard) {
      creatingPathRef.current = null;
      return;
    }
    if (
      isCreateBoardPending ||
      creatingPathRef.current === selected.path ||
      failedCreatePathsRef.current.has(selected.path)
    ) {
      return;
    }
    creatingPathRef.current = selected.path;
    createBoard(
      { name: selected.name, project_id: selected.path },
      {
        onError: () => {
          failedCreatePathsRef.current.add(selected.path);
          if (creatingPathRef.current === selected.path) {
            creatingPathRef.current = null;
          }
          displayErrorToast(t(I18nKey.ERROR$GENERIC));
        },
      },
    );
  }, [
    boardsQuery.isLoading,
    createBoard,
    isCreateBoardPending,
    selectedBoard,
    t,
    workspace.selected,
  ]);

  const isCreatingBoard =
    Boolean(workspace.selected) &&
    !board &&
    (isCreateBoardPending ||
      boardsQuery.isFetching ||
      boardsQuery.isLoading ||
      boardQuery.isLoading);

  return (
    <main
      data-testid="kanban-page"
      aria-label={t(I18nKey.KANBAN$NAV)}
      className={kanbanPageShellClassName}
    >
      <header className="mb-3 flex h-9 shrink-0 items-center justify-between gap-3">
        <KanbanWorkspacePicker
          workspaces={workspace.workspaces}
          parents={workspace.parents}
          workspaceParents={workspace.workspaceParents}
          selected={workspace.selected}
          isLoading={workspace.isLoading}
          listError={workspace.listError}
          onChange={workspace.setSelected}
        />
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
          <StyledTooltip content={t(I18nKey.PROJECT_INIT$NAV)}>
            <button
              type="button"
              data-testid="kanban-new-project"
              disabled={!workspace.selected}
              aria-label={t(I18nKey.PROJECT_INIT$NAV)}
              onClick={() => navigate(PROJECT_INIT_PATH)}
              className="inline-flex h-8 w-8 items-center justify-center rounded-md text-[var(--oh-muted)] hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--oh-foreground)] disabled:cursor-not-allowed disabled:opacity-40"
            >
              <FolderPlus className="h-4 w-4" aria-hidden />
            </button>
          </StyledTooltip>
        </div>
      </header>

      {!workspace.selected && !workspace.isLoading ? (
        <div
          data-testid="kanban-empty"
          className={extensionModuleEmptyStateClassName}
        >
          <p className="text-sm font-medium text-white">
            {t(I18nKey.KANBAN$NO_WORKSPACE)}
          </p>
          <p className="mt-2 text-sm text-tertiary-light">
            {t(I18nKey.KANBAN$NO_WORKSPACE_HINT)}
          </p>
        </div>
      ) : null}

      {isCreatingBoard ? (
        <p
          data-testid="kanban-creating"
          className="text-sm text-tertiary-light"
        >
          {t(I18nKey.KANBAN$CREATING_BOARD)}
        </p>
      ) : null}

      {board ? (
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden lg:flex-row">
          <div className="min-h-0 min-w-0 flex-1 overflow-hidden">
            {view === KANBAN_VIEW_BOARD ? (
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
            ) : (
              <KanbanList board={board} onSelectCard={setSelectedCard} />
            )}
          </div>
          {selectedFromBoard ? (
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
          ) : null}
        </div>
      ) : null}
    </main>
  );
}
