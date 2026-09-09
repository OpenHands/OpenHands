import React from "react";
import { useTranslation } from "react-i18next";
import {
  Columns3,
  FolderPlus,
  GitCompare,
  Link2,
  List,
  Rows3,
} from "lucide-react";
import {
  KANBAN_VIEW_BOARD,
  KANBAN_VIEW_LIST,
  PROJECT_INIT_PATH,
} from "#/api/kanban-service/kanban-constants";
import type {
  KanbanBoard,
  KanbanCard,
} from "#/api/kanban-service/kanban-types";
import { CardDetailPanel } from "#/components/features/kanban/card-detail-panel";
import { CostSummary } from "#/components/features/kanban/cost-summary";
import { KanbanBoardView } from "#/components/features/kanban/kanban-board";
import { KanbanList } from "#/components/features/kanban/kanban-list";
import { KanbanMapSourceModal } from "#/components/features/kanban/kanban-map-source-modal";
import { KanbanReconcileModal } from "#/components/features/kanban/kanban-reconcile-modal";
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
  useKanbanBoardsDetail,
  useMoveKanbanCard,
  useUpdateKanbanCard,
} from "#/hooks/query/use-kanban";
import { useKanbanWorkspace } from "#/hooks/use-kanban-workspace";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { I18nKey } from "#/i18n/declaration";
import { useKanbanSyncStore } from "#/stores/kanban-sync-store";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";
import { kanbanPageShellClassName } from "#/utils/kanban-page-layout-classes";
import {
  buildBoardSwimlanes,
  buildWorkspaceSwimlanes,
  costsFromBoards,
  flattenBoardCards,
} from "#/utils/kanban-swimlanes";

type KanbanView = typeof KANBAN_VIEW_BOARD | typeof KANBAN_VIEW_LIST;

export default function KanbanPage() {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const [view, setView] = React.useState<KanbanView>(KANBAN_VIEW_BOARD);
  const [selectedCard, setSelectedCard] = React.useState<KanbanCard | null>(
    null,
  );
  const [addingLane, setAddingLane] = React.useState(false);
  const [laneDraft, setLaneDraft] = React.useState("");
  const creatingPathRef = React.useRef<string | null>(null);

  const workspace = useKanbanWorkspace();
  const boardsQuery = useKanbanBoards();
  const selectedBoard = boardForWorkspace(
    boardsQuery.data ?? [],
    workspace.selected?.path ?? null,
  );
  const selectedBoardId = workspace.isAllWorkspaces
    ? null
    : (selectedBoard?.id ?? null);
  const boardQuery = useKanbanBoard(selectedBoardId);
  const costsQuery = useKanbanBoardCosts(selectedBoardId);
  const workspaceBoards = (boardsQuery.data ?? []).filter((summary) =>
    workspace.workspaces.some((item) => item.path === summary.project_id),
  );
  const combinedQueries = useKanbanBoardsDetail(
    workspace.isAllWorkspaces ? workspaceBoards.map((item) => item.id) : [],
  );
  const { mutate: createBoard, isPending: isCreateBoardPending } =
    useCreateKanbanBoard();
  const failedCreatePathsRef = React.useRef(new Set<string>());
  const mutationBoardId = selectedBoardId ?? "";
  const createCard = useCreateKanbanCard(mutationBoardId);
  const createColumn = useCreateKanbanColumn(mutationBoardId);
  const moveCard = useMoveKanbanCard(mutationBoardId);
  const updateCard = useUpdateKanbanCard(mutationBoardId);
  const deleteCard = useDeleteKanbanCard(mutationBoardId);

  const lanes = useKanbanSyncStore((state) => state.lanes);
  const collapsedLaneIds = useKanbanSyncStore(
    (state) => state.collapsedLaneIds,
  );
  const toggleLaneCollapsed = useKanbanSyncStore(
    (state) => state.toggleLaneCollapsed,
  );
  const addLane = useKanbanSyncStore((state) => state.addLane);
  const conflicts = useKanbanSyncStore((state) => state.conflicts);
  const setMappingOpen = useKanbanSyncStore((state) => state.setMappingOpen);
  const setReconcileOpen = useKanbanSyncStore(
    (state) => state.setReconcileOpen,
  );
  const detectAfterLocalChange = useKanbanSyncStore(
    (state) => state.detectAfterLocalChange,
  );
  const sources = useKanbanSyncStore((state) => state.sources);

  const combinedBoards = combinedQueries
    .map((query) => query.data)
    .filter((board): board is KanbanBoard => Boolean(board));
  const board = workspace.isAllWorkspaces
    ? (combinedBoards[0] ?? null)
    : (boardQuery.data ?? null);
  const visibleBoards = workspace.isAllWorkspaces
    ? combinedBoards
    : board
      ? [board]
      : [];
  const selectedFromBoard =
    visibleBoards
      .flatMap((item) => flattenBoardCards(item))
      .find((card) => card.id === selectedCard?.id) ?? null;
  const localCards = visibleBoards.flatMap(flattenBoardCards);
  const swimlanes = workspace.isAllWorkspaces
    ? buildWorkspaceSwimlanes({
        boards: combinedBoards.map((item) => ({
          board: item,
          workspaceName:
            workspace.workspaces.find((entry) => entry.path === item.project_id)
              ?.name ?? item.name,
        })),
        lanes,
      })
    : board
      ? buildBoardSwimlanes({ board, lanes })
      : [];
  const combinedCosts = workspace.isAllWorkspaces
    ? costsFromBoards(combinedBoards)
    : costsQuery.data;

  React.useEffect(() => {
    setSelectedCard(null);
  }, [workspace.selected?.path, workspace.isAllWorkspaces]);

  React.useEffect(() => {
    const selected = workspace.selected;
    if (workspace.isAllWorkspaces || !selected?.path || boardsQuery.isLoading) {
      return;
    }
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
    workspace.isAllWorkspaces,
    workspace.selected,
  ]);

  const isCreatingBoard =
    !workspace.isAllWorkspaces &&
    Boolean(workspace.selected) &&
    !board &&
    (isCreateBoardPending ||
      boardsQuery.isFetching ||
      boardsQuery.isLoading ||
      boardQuery.isLoading);

  const showBoard =
    Boolean(board) || (workspace.isAllWorkspaces && combinedBoards.length > 0);
  const defaultBoardId = selectedBoardId ?? combinedBoards[0]?.id ?? null;

  const handleAddCard = (
    columnId: string,
    title: string,
    laneId: string | null,
  ) => {
    createCard.mutate(
      { columnId, payload: { title, lane_id: laneId } },
      {
        onSuccess: () => {
          if (defaultBoardId) {
            detectAfterLocalChange(defaultBoardId, localCards);
          }
        },
      },
    );
  };

  const handleMoveCard = (
    cardId: string,
    columnId: string,
    position: number,
    laneId: string | null,
  ) => {
    moveCard.mutate({
      cardId,
      payload: { column_id: columnId, position },
    });
    if (laneId !== undefined) {
      updateCard.mutate({ cardId, payload: { lane_id: laneId } });
    }
  };

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
          isAllWorkspaces={workspace.isAllWorkspaces}
          isLoading={workspace.isLoading}
          listError={workspace.listError}
          onChange={workspace.setSelected}
          onSelectAll={workspace.setAllWorkspaces}
        />
        <div className="flex shrink-0 items-center gap-2">
          {conflicts.length > 0 ? (
            <button
              type="button"
              data-testid="kanban-conflict-banner"
              onClick={() => setReconcileOpen(true)}
              className="rounded-md bg-amber-500/10 px-2 py-1 text-[11px] text-amber-200 hover:bg-amber-500/20"
            >
              {t(I18nKey.KANBAN$CONFLICT_BANNER, { count: conflicts.length })}
            </button>
          ) : null}
          {combinedCosts ? <CostSummary costs={combinedCosts} /> : null}
          {showBoard ? (
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
          <StyledTooltip content={t(I18nKey.KANBAN$MAP_SOURCE)}>
            <button
              type="button"
              data-testid="kanban-map-source"
              disabled={!showBoard}
              aria-label={t(I18nKey.KANBAN$MAP_SOURCE)}
              onClick={() => setMappingOpen(true)}
              className="inline-flex h-8 w-8 items-center justify-center rounded-md text-[var(--oh-muted)] hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--oh-foreground)] disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Link2 className="h-4 w-4" aria-hidden />
            </button>
          </StyledTooltip>
          <StyledTooltip content={t(I18nKey.KANBAN$RECONCILE)}>
            <button
              type="button"
              data-testid="kanban-reconcile"
              disabled={!showBoard || sources.length === 0}
              aria-label={t(I18nKey.KANBAN$RECONCILE)}
              onClick={() => {
                if (defaultBoardId) {
                  useKanbanSyncStore
                    .getState()
                    .reconcileBoard(defaultBoardId, localCards);
                }
                setReconcileOpen(true);
              }}
              className="inline-flex h-8 w-8 items-center justify-center rounded-md text-[var(--oh-muted)] hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--oh-foreground)] disabled:cursor-not-allowed disabled:opacity-40"
            >
              <GitCompare className="h-4 w-4" aria-hidden />
            </button>
          </StyledTooltip>
          {board && !workspace.isAllWorkspaces ? (
            addingLane ? (
              <form
                onSubmit={(event) => {
                  event.preventDefault();
                  const name = laneDraft.trim();
                  if (!name) return;
                  addLane({ boardId: board.id, name });
                  setLaneDraft("");
                  setAddingLane(false);
                }}
              >
                <input
                  data-testid="kanban-swimlane-name"
                  value={laneDraft}
                  onChange={(event) => setLaneDraft(event.target.value)}
                  onBlur={() => {
                    if (!laneDraft.trim()) setAddingLane(false);
                  }}
                  placeholder={t(I18nKey.KANBAN$NEW_SWIMLANE_NAME)}
                  aria-label={t(I18nKey.KANBAN$ADD_SWIMLANE)}
                  className="h-8 w-36 rounded-md border border-[var(--oh-border)] bg-transparent px-2 text-xs text-[var(--oh-foreground)]"
                />
              </form>
            ) : (
              <StyledTooltip content={t(I18nKey.KANBAN$ADD_SWIMLANE)}>
                <button
                  type="button"
                  data-testid="kanban-add-swimlane"
                  aria-label={t(I18nKey.KANBAN$ADD_SWIMLANE)}
                  onClick={() => setAddingLane(true)}
                  className="inline-flex h-8 w-8 items-center justify-center rounded-md text-[var(--oh-muted)] hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--oh-foreground)]"
                >
                  <Rows3 className="h-4 w-4" aria-hidden />
                </button>
              </StyledTooltip>
            )
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

      {!workspace.selected &&
      !workspace.isAllWorkspaces &&
      !workspace.isLoading ? (
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

      {showBoard && board ? (
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden lg:flex-row">
          <div className="min-h-0 min-w-0 flex-1 overflow-hidden">
            {view === KANBAN_VIEW_BOARD ? (
              <KanbanBoardView
                board={board}
                boards={visibleBoards}
                costs={combinedCosts}
                lanes={swimlanes}
                collapsedLaneIds={collapsedLaneIds}
                onToggleLane={toggleLaneCollapsed}
                onSelectCard={setSelectedCard}
                onAddCard={handleAddCard}
                onMoveCard={handleMoveCard}
                onAddColumn={
                  workspace.isAllWorkspaces
                    ? undefined
                    : (name) => createColumn.mutate({ name })
                }
              />
            ) : (
              <KanbanList
                board={board}
                lanes={swimlanes}
                onSelectCard={setSelectedCard}
              />
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

      <KanbanMapSourceModal
        boards={visibleBoards.map((item) => ({
          board: item,
          name:
            workspace.workspaces.find((entry) => entry.path === item.project_id)
              ?.name ?? item.name,
        }))}
        localCards={localCards}
        defaultBoardId={defaultBoardId}
      />
      <KanbanReconcileModal
        boards={visibleBoards}
        localCards={localCards}
        onUpdateCard={(cardId, payload) =>
          updateCard.mutate({ cardId, payload })
        }
        onCreateCard={(columnId, payload) =>
          createCard.mutate({ columnId, payload })
        }
      />
    </main>
  );
}
