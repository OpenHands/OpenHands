import React from "react";
import { Tooltip } from "@heroui/react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useNavigation } from "#/context/navigation-context";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useBackendScopedPath } from "#/hooks/use-backend-scoped-path";
import { usePaginatedConversations } from "#/hooks/query/use-paginated-conversations";
import { useStartTasks } from "#/hooks/query/use-start-tasks";
import { useDeleteConversation } from "#/hooks/mutation/use-delete-conversation";
import { useUnifiedPauseConversation } from "#/hooks/mutation/use-unified-stop-conversation";
import { ConfirmArchiveModal } from "./confirm-archive-modal";
import { ConfirmDeleteModal } from "./confirm-delete-modal";
import { ConfirmStopModal } from "./confirm-stop-modal";
import { NavigationLink } from "#/components/shared/navigation-link";
import { ExitConversationModal } from "./exit-conversation-modal";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { Provider } from "#/types/settings";
import { useUpdateConversation } from "#/hooks/mutation/use-update-conversation";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { isExecutionActive } from "#/utils/status";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useIsCreatingConversation } from "#/hooks/use-is-creating-conversation";
import { ConversationCard } from "./conversation-card/conversation-card";
import { ConversationCardPreview } from "./conversation-card/conversation-card-preview";
import { StartTaskCard } from "./start-task-card/start-task-card";
import { ConversationCardSkeleton } from "./conversation-card/conversation-card-skeleton";
import { CompactConversationRow } from "./compact-conversation-row";
import { useConversationPanelPreferencesStore } from "#/stores/conversation-panel-preferences-store";
import { cn } from "#/utils/utils";
import { ConversationPanelFilterMenu } from "./conversation-panel-filter-menu";
import { ConversationPanelNewThreadPicker } from "./conversation-panel-new-thread-picker";
import { ConversationGroupFolderList } from "./conversation-group-folder-list";
import { ConversationPanelPinnedSection } from "./conversation-panel-pinned-section";
import {
  applyAutomationConversationFilter,
  applyGroupFolderOrder,
  collectAutomationNameFacets,
  filterOutPinnedConversations,
  getGroupDiscoveryConversationIds,
  groupConversations,
  MAX_PAGES_PER_LOAD_MORE_CLICK,
  resolvePinnedConversations,
  sortConversationsByField,
  type ConversationGroupLaunch,
} from "./conversation-panel-list-helpers";
import { useArchivedConversationsStore } from "#/stores/archived-conversations-store";
import { usePinnedConversationsStore } from "#/stores/pinned-conversations-store";

interface ConversationPanelProps {
  onClose?: () => void;
  /**
   * Render a minimal icon-only variant of each conversation row (used by the
   * collapsed sidebar). Each row is a single status dot with a hover preview
   * containing the full card content.
   */
  compact?: boolean;
}

const noop = () => {};

const EMPTY_PINNED_CONVERSATION_IDS: readonly string[] = [];

const ONE_HOUR_MS = 60 * 60 * 1000;

const partitionByCutoff = <T extends { updated_at: string }>(
  items: readonly T[],
): { recent: T[]; older: T[] } => {
  // The cutoff is intentionally relative to "now" each time the list is
  // recomputed, so conversations naturally age into the older bucket as the
  // conversations query refreshes.
  const cutoff = Date.now() - ONE_HOUR_MS;
  const recent: T[] = [];
  const older: T[] = [];
  for (const item of items) {
    const updatedAt = item.updated_at ? Date.parse(item.updated_at) : NaN;
    // Missing or unparseable timestamps stay in the "recent" bucket so we
    // do not accidentally hide them behind the older-conversations toggle.
    if (Number.isFinite(updatedAt) && updatedAt < cutoff) {
      older.push(item);
    } else {
      recent.push(item);
    }
  }
  return { recent, older };
};

export function ConversationPanel({
  onClose,
  compact = false,
}: ConversationPanelProps) {
  const { t } = useTranslation("openhands");
  const { conversationId: currentConversationId, navigate } = useNavigation();
  const { backend: activeBackend } = useActiveBackend();
  const backendScopedPath = useBackendScopedPath();
  // Click-outside is only relevant in the legacy drawer mode where an
  // onClose handler is provided. When the panel is rendered inline (e.g.
  // as the always-visible conversation list pane), clicking outside should
  // not dismiss the list, so we pass a no-op callback in that case.
  const ref = useClickOutsideElement<HTMLDivElement>(onClose ?? noop);

  const [confirmDeleteModalVisible, setConfirmDeleteModalVisible] =
    React.useState(false);
  const [confirmArchiveModalVisible, setConfirmArchiveModalVisible] =
    React.useState(false);
  const [confirmStopModalVisible, setConfirmStopModalVisible] =
    React.useState(false);
  const [
    confirmExitConversationModalVisible,
    setConfirmExitConversationModalVisible,
  ] = React.useState(false);
  const [confirmDeleteAllVisible, setConfirmDeleteAllVisible] =
    React.useState(false);
  const showOlderConversations = useConversationPanelPreferencesStore(
    (state) => state.showOlderConversations,
  );
  const showArchivedConversations = useConversationPanelPreferencesStore(
    (state) => state.showArchivedConversations,
  );
  const toggleShowArchivedConversations = useConversationPanelPreferencesStore(
    (state) => state.toggleShowArchivedConversations,
  );
  const toggleShowOlderConversations = useConversationPanelPreferencesStore(
    (state) => state.toggleShowOlderConversations,
  );
  const showRepoBranchMetadata = useConversationPanelPreferencesStore(
    (state) => state.showRepoBranchMetadata,
  );
  const toggleShowRepoBranchMetadata = useConversationPanelPreferencesStore(
    (state) => state.toggleShowRepoBranchMetadata,
  );
  const showLlmProfiles = useConversationPanelPreferencesStore(
    (state) => state.showLlmProfiles,
  );
  const toggleShowLlmProfiles = useConversationPanelPreferencesStore(
    (state) => state.toggleShowLlmProfiles,
  );
  const showTagsMetadata = useConversationPanelPreferencesStore(
    (state) => state.showTagsMetadata,
  );
  const toggleShowTagsMetadata = useConversationPanelPreferencesStore(
    (state) => state.toggleShowTagsMetadata,
  );
  const showHoverMetadata = useConversationPanelPreferencesStore(
    (state) => state.showHoverMetadata,
  );
  const toggleShowHoverMetadata = useConversationPanelPreferencesStore(
    (state) => state.toggleShowHoverMetadata,
  );
  const organizeMode = useConversationPanelPreferencesStore(
    (state) => state.organizeMode,
  );
  const setOrganizeMode = useConversationPanelPreferencesStore(
    (state) => state.setOrganizeMode,
  );
  const conversationSort = useConversationPanelPreferencesStore(
    (state) => state.conversationSort,
  );
  const setConversationSort = useConversationPanelPreferencesStore(
    (state) => state.setConversationSort,
  );
  const threadScope = useConversationPanelPreferencesStore(
    (state) => state.threadScope,
  );
  const setThreadScope = useConversationPanelPreferencesStore(
    (state) => state.setThreadScope,
  );
  const automationFilterMode = useConversationPanelPreferencesStore(
    (state) => state.automationFilterMode,
  );
  const setAutomationFilterMode = useConversationPanelPreferencesStore(
    (state) => state.setAutomationFilterMode,
  );
  const selectedAutomationNames = useConversationPanelPreferencesStore(
    (state) => state.selectedAutomationNames,
  );
  const toggleAutomationName = useConversationPanelPreferencesStore(
    (state) => state.toggleAutomationName,
  );
  const groupFolderOrder = useConversationPanelPreferencesStore(
    (state) => state.groupFolderOrder,
  );
  const setGroupFolderOrder = useConversationPanelPreferencesStore(
    (state) => state.setGroupFolderOrder,
  );
  const [filterMenuOpen, setFilterMenuOpen] = React.useState(false);
  const [isListScrolled, setIsListScrolled] = React.useState(false);
  const filterMenuRef = useClickOutsideElement<HTMLDivElement>(() => {
    setFilterMenuOpen(false);
  });
  const [collapsedGroupIds, setCollapsedGroupIds] = React.useState<
    ReadonlySet<string>
  >(() => new Set());
  const [expandedGroupPreviewIds, setExpandedGroupPreviewIds] = React.useState<
    ReadonlySet<string>
  >(() => new Set());
  const [expandedPinnedPreview, setExpandedPinnedPreview] =
    React.useState(false);

  const pinnedIds = usePinnedConversationsStore(
    (state) =>
      state.pinsByBackendId[activeBackend.id] ?? EMPTY_PINNED_CONVERSATION_IDS,
  );
  const togglePin = usePinnedConversationsStore((state) => state.togglePin);
  const unpinConversation = usePinnedConversationsStore(
    (state) => state.unpinConversation,
  );
  const pruneMissingPinnedConversations = usePinnedConversationsStore(
    (state) => state.pruneMissingConversations,
  );
  const archivedIds = useArchivedConversationsStore(
    (state) =>
      state.archivesByBackendId[activeBackend.id] ??
      EMPTY_PINNED_CONVERSATION_IDS,
  );
  const archiveConversation = useArchivedConversationsStore(
    (state) => state.archiveConversation,
  );
  const archivedIdSet = React.useMemo(
    () => new Set(archivedIds),
    [archivedIds],
  );
  const removeArchivedConversation = useArchivedConversationsStore(
    (state) => state.removeArchivedConversation,
  );

  const toggleGroupCollapsed = React.useCallback((groupId: string) => {
    setCollapsedGroupIds((prev) => {
      const next = new Set(prev);
      if (next.has(groupId)) {
        next.delete(groupId);
      } else {
        next.add(groupId);
      }
      return next;
    });
  }, []);

  const toggleGroupPreviewExpanded = React.useCallback((groupId: string) => {
    setExpandedGroupPreviewIds((prev) => {
      const next = new Set(prev);
      if (next.has(groupId)) {
        next.delete(groupId);
      } else {
        next.add(groupId);
      }
      return next;
    });
  }, []);

  React.useEffect(() => {
    if (organizeMode !== "grouped") {
      setCollapsedGroupIds(new Set());
      setExpandedGroupPreviewIds(new Set());
    }
  }, [organizeMode]);

  const scrollContainerRef = React.useRef<HTMLDivElement>(null);

  const [selectedConversationId, setSelectedConversationId] = React.useState<
    string | null
  >(null);
  const [selectedConversationTitle, setSelectedConversationTitle] =
    React.useState<string | null>(null);
  const [openContextMenuId, setOpenContextMenuId] = React.useState<
    string | null
  >(null);

  const {
    data,
    isLoading,
    isFetched,
    hasNextPage,
    isFetching,
    isFetchingNextPage,
    fetchNextPage,
  } = usePaginatedConversations();

  // Fetch in-progress start tasks
  const { data: startTasks } = useStartTasks();

  // Deduped, archive-unaware collection of every conversation currently loaded
  // from the backend. Bulk actions like "Delete all" must use this list so
  // hiding archived rows from the UI never shrinks what gets deleted.
  const allLoadedConversations = React.useMemo(() => {
    const all = data?.pages.flatMap((page) => page.items) ?? [];
    // The 10s background refetch re-fetches every loaded page with the
    // `UPDATED_AT_DESC` cursor. If a conversation's `updated_at` shifts between
    // page fetches, a later page can overlap an earlier one and surface the
    // same conversation twice. Dedupe by id (keeping the first/freshest copy)
    // so the rendered count reflects real growth and React keys stay unique.
    const seen = new Set<string>();
    return all.filter((conversation) => {
      if (seen.has(conversation.id)) {
        return false;
      }
      seen.add(conversation.id);
      return true;
    });
  }, [data]);

  // Grouped pagination is folder-oriented. Record the first backend page for
  // every conversation so later pages can introduce new folders without
  // mutating the contents of folders that are already visible.
  //
  // Known limitation: the mapping is recomputed from `data.pages` on the 10s
  // background refetch, so a conversation whose `updated_at` shifts can move
  // to an earlier page and change its recorded discovery page. The blast
  // radius is only which rows a collapsed preview shows (expanding a folder
  // always reveals every loaded conversation); pinning discovery pages across
  // refetches is not worth the extra bookkeeping today.
  const conversationPageById = React.useMemo(() => {
    const pageById = new Map<string, number>();
    data?.pages.forEach((page, pageIndex) => {
      page.items.forEach((conversation) => {
        if (!pageById.has(conversation.id)) {
          pageById.set(conversation.id, pageIndex);
        }
      });
    });
    return pageById;
  }, [data]);

  // Display collection: same loaded pages, with archived rows filtered out
  // unless the user has opted into "Show archived".
  const conversations = React.useMemo(() => {
    if (showArchivedConversations) {
      return allLoadedConversations;
    }
    return allLoadedConversations.filter(
      (conversation) => !archivedIdSet.has(conversation.id),
    );
  }, [allLoadedConversations, archivedIdSet, showArchivedConversations]);

  // Facets derive from the unfiltered list so the automation-name rows in the
  // filter menu don't vanish while a narrowing selection is active.
  const automationNameFacets = React.useMemo(
    () => collectAutomationNameFacets(conversations),
    [conversations],
  );

  const automationFilteredConversations = React.useMemo(
    () =>
      applyAutomationConversationFilter(
        conversations,
        automationFilterMode,
        selectedAutomationNames,
        automationNameFacets,
      ),
    [
      automationFilterMode,
      automationNameFacets,
      conversations,
      selectedAutomationNames,
    ],
  );

  const pinnedConversations = React.useMemo(
    () => resolvePinnedConversations(pinnedIds, conversations),
    [conversations, pinnedIds],
  );

  React.useEffect(() => {
    if (!isFetched) {
      return;
    }
    // Prune pins against the unfiltered loaded pages so archived-but-still-
    // pinned rows are not treated as missing. Archived IDs are intentionally
    // not pruned here — pagination would otherwise drop archives that are not
    // on the currently loaded pages and let them reappear in the list.
    const loadedIds =
      data?.pages.flatMap((page) => page.items.map((item) => item.id)) ?? [];
    pruneMissingPinnedConversations(activeBackend.id, loadedIds);
  }, [activeBackend.id, data, isFetched, pruneMissingPinnedConversations]);

  React.useEffect(() => {
    if (pinnedIds.length === 0) {
      setExpandedPinnedPreview(false);
    }
  }, [pinnedIds.length]);

  const scopedConversations = React.useMemo(() => {
    // The pinned section intentionally bypasses the automation filter (same
    // exemption the thread scope has): a pin is an explicit user override.
    const scopeFiltered =
      threadScope === "relevant"
        ? automationFilteredConversations.filter((c) =>
            isExecutionActive(c.execution_status),
          )
        : automationFilteredConversations;

    // In the expanded panel, pinned conversations should only appear inside
    // the dedicated pinned section (not duplicated in grouped/flat lists).
    if (compact) {
      return scopeFiltered;
    }

    return filterOutPinnedConversations(scopeFiltered, pinnedIds);
  }, [automationFilteredConversations, compact, pinnedIds, threadScope]);

  const { recent: recentScoped, older: olderScoped } = React.useMemo(
    () => partitionByCutoff(scopedConversations),
    [scopedConversations],
  );

  // Sort the full visible set as one list. The recent/older partition is
  // still computed (it gates the "Show older" toggle and "Load more"
  // visibility), but the rendering must not use it as a visual boundary —
  // when sorting by `created`, a stale-but-recently-touched conversation
  // would otherwise land in `recent` and render above an actually-newer-
  // by-`created_at` conversation sitting in `older`.
  const sortedVisibleConversations = React.useMemo(() => {
    const visible = showOlderConversations
      ? [...recentScoped, ...olderScoped]
      : recentScoped;
    return sortConversationsByField(visible, conversationSort);
  }, [recentScoped, olderScoped, showOlderConversations, conversationSort]);

  const groupLabels = React.useMemo(
    () => ({
      emptyWorkspace: t(I18nKey.CONVERSATION_PANEL$NO_WORKSPACE),
      emptyRepository: t(I18nKey.CONVERSATION_PANEL$NO_REPOSITORY),
    }),
    [t],
  );

  const groupedSourceConversations = React.useMemo(() => {
    if (compact || organizeMode !== "grouped") {
      return null;
    }
    // Use the unsorted partitions: groupConversations sorts each bucket
    // internally by `sortField`, so pre-sorting the merged input is wasted
    // work in grouped mode (the per-group sort overrides any global order).
    return [...recentScoped, ...(showOlderConversations ? olderScoped : [])];
  }, [
    compact,
    olderScoped,
    organizeMode,
    recentScoped,
    showOlderConversations,
  ]);

  const conversationGroups = React.useMemo(() => {
    if (!groupedSourceConversations) {
      return null;
    }
    // Keep every loaded conversation in the group model. Folder discovery only
    // freezes the collapsed preview — expanding a folder must reach later-page
    // rows for that same workspace/repo.
    return groupConversations(
      groupedSourceConversations,
      activeBackend.kind,
      conversationSort,
      groupLabels,
    );
  }, [
    activeBackend.kind,
    conversationSort,
    groupLabels,
    groupedSourceConversations,
  ]);

  const groupDiscoveryConversationIds = React.useMemo(() => {
    if (!groupedSourceConversations) {
      return null;
    }
    return getGroupDiscoveryConversationIds(
      groupedSourceConversations,
      conversationPageById,
      activeBackend.kind,
      { forceIncludeConversationId: currentConversationId },
    );
  }, [
    activeBackend.kind,
    conversationPageById,
    currentConversationId,
    groupedSourceConversations,
  ]);

  const orderedConversationGroups = React.useMemo(() => {
    if (!conversationGroups) {
      return null;
    }
    return applyGroupFolderOrder(conversationGroups, groupFolderOrder);
  }, [conversationGroups, groupFolderOrder]);

  const conversationGroupIds = React.useMemo(
    () => conversationGroups?.map((group) => group.id) ?? [],
    [conversationGroups],
  );

  const compactVisibleConversations = React.useMemo(
    () =>
      sortConversationsByField(
        recentScoped.filter((conversation) =>
          isExecutionActive(conversation.execution_status),
        ),
        conversationSort,
      ),
    [conversationSort, recentScoped],
  );

  const visibleFlatCount = sortedVisibleConversations.length;

  const visibleGroupCount = orderedConversationGroups?.length ?? 0;

  const listIsEffectivelyEmpty =
    organizeMode === "grouped" && !compact
      ? visibleGroupCount === 0
      : visibleFlatCount === 0;

  // Attribution is exact: the automation filter step itself produced zero
  // rows out of a non-empty loaded set (not merely threadScope/older-cutoff
  // effects). Used to pick the empty-state message.
  const emptyDueToAutomationFilter =
    listIsEffectivelyEmpty &&
    automationFilterMode !== "all" &&
    conversations.length > 0 &&
    automationFilteredConversations.length === 0;

  // Grouped pagination prefers discovering another folder; chronological
  // pagination succeeds when another row appears. Pages that only deepen
  // already-visible folders are not success for the grouped control — the
  // driver walks past them (bounded by the per-click page cap) so a single
  // click can still reach a folder hiding behind deepen-only pages.
  const visibleCount =
    organizeMode === "grouped" && !compact
      ? visibleGroupCount
      : visibleFlatCount;
  const loadedPageCount = data?.pages.length ?? 0;

  // KNOWN ISSUE (unresolved as of 2026-05-29): users still report that the
  // sidebar "Load more" sometimes requires two clicks before new conversations
  // appear. The mitigation below (dedupe by id in `conversations`, plus the
  // floor-tracking driver that keeps fetching until the visible count grows)
  // reduced but did NOT fully eliminate the symptom in manual testing. Likely
  // remaining suspects to investigate next: the agent-server cursor pagination
  // returning an overlapping/short page under `UPDATED_AT_DESC` while the 10s
  // `refetchInterval` reorders pages (see `usePaginatedConversations`), or a
  // React Query state lag where `hasNextPage`/`isFetching` settle a render
  // after the click. If you pick this up, reproduce against a backend with
  // >40 conversations and watch the `/api/conversations/search` cursors.
  //
  // Robust "Load more" driver. A single click can fail to surface new rows for
  // two reasons: (1) `fetchNextPage()` is silently dropped while the 10s
  // background refetch is in flight, and (2) a fetched page can yield zero
  // *visible* rows (filtered out by the active scope, or deduped as overlap),
  // so the list does not appear to grow. We capture floors at click time and
  // keep fetching — once idle — until the visible count grows, the grouped
  // per-click page cap is hit, or pages run out. `loadedPageCount` keeps the
  // driver advancing when a fetched page contains only folders that were
  // already discovered.
  const [loadMoreFloor, setLoadMoreFloor] = React.useState<number | null>(null);
  const [loadMorePageFloor, setLoadMorePageFloor] = React.useState<
    number | null
  >(null);
  const visibleCountRef = React.useRef(visibleCount);
  visibleCountRef.current = visibleCount;
  const loadedPageCountRef = React.useRef(loadedPageCount);
  loadedPageCountRef.current = loadedPageCount;

  const clearLoadMoreRequest = React.useCallback(() => {
    setLoadMoreFloor(null);
    setLoadMorePageFloor(null);
  }, []);

  const requestLoadMore = React.useCallback(() => {
    if (hasNextPage) {
      setLoadMoreFloor(visibleCountRef.current);
      setLoadMorePageFloor(loadedPageCountRef.current);
    }
  }, [hasNextPage]);

  React.useEffect(() => {
    if (loadMoreFloor === null) {
      return;
    }
    // Goal met: the visible list grew past where it was when the user clicked.
    if (visibleCount > loadMoreFloor) {
      clearLoadMoreRequest();
      return;
    }
    // Hard cap (grouped only): pages that merely deepen already-visible
    // folders are walked past — a new folder may sit right behind them — but
    // never unbounded many, so one click cannot drain the whole cursor.
    // Chronological mode keeps its pre-existing behavior: fetch until a
    // visible row appears or pages run out.
    if (
      organizeMode === "grouped" &&
      !compact &&
      loadMorePageFloor != null &&
      loadedPageCount >= loadMorePageFloor + MAX_PAGES_PER_LOAD_MORE_CLICK
    ) {
      clearLoadMoreRequest();
      return;
    }
    // Wait for any in-flight fetch (including the background refetch) to settle
    // before evaluating `hasNextPage`; React Query may transiently clear that
    // flag while replacing the last page.
    if (isFetching || isFetchingNextPage) {
      return;
    }
    // Nothing more to fetch — stop waiting even if the list did not grow.
    if (!hasNextPage) {
      clearLoadMoreRequest();
      return;
    }
    fetchNextPage();
  }, [
    clearLoadMoreRequest,
    compact,
    loadMoreFloor,
    loadMorePageFloor,
    visibleCount,
    loadedPageCount,
    organizeMode,
    hasNextPage,
    isFetching,
    isFetchingNextPage,
    fetchNextPage,
  ]);

  const isLoadingMore = loadMoreFloor !== null || isFetchingNextPage;

  const { mutate: deleteConversation, mutateAsync: deleteConversationAsync } =
    useDeleteConversation();
  const { mutate: pauseConversation } = useUnifiedPauseConversation();
  const { mutate: updateConversation } = useUpdateConversation();

  // The next page of conversations is loaded only via the explicit "Load
  // more" link rendered at the end of the list — there is no scroll-driven
  // pagination, which previously caused the panel to feel like it had stray
  // scrollable space at the bottom.
  const olderHidden = olderScoped.length > 0 && !showOlderConversations;
  // Compact mode also hides "Load more" — paginating into stale conversations
  // contradicts the "active only" intent of the icon rail.
  // Availability tracks backend exhaustion, never visible emptiness: a
  // client-side filter (archiving, the automation filter) can hide every row
  // of the loaded pages, and hiding the control there would strand the
  // remaining pages behind an empty-state message with no way forward.
  // `requestLoadMore`'s floor driver keeps paging until a visible row
  // appears, so a single click walks past pages that are entirely filtered.
  const showLoadMore = !!hasNextPage && !olderHidden && !compact;

  const { mutate: createConversation } = useCreateConversation();
  const isCreatingConversationFlow = useIsCreatingConversation();

  const launchFromGroup = React.useCallback(
    (launch: ConversationGroupLaunch) => {
      if (isCreatingConversationFlow) return;
      createConversation(
        {
          workingDir: launch.workingDir,
          repository: launch.repository,
          entryPoint: "sidebar_relaunch_project",
        },
        {
          onSuccess: (data) => {
            navigate(`/conversations/${data.conversation_id}`);
          },
        },
      );
    },
    [createConversation, isCreatingConversationFlow, navigate],
  );

  const handleDeleteProject = React.useCallback(
    (conversationId: string, title: string) => {
      setConfirmDeleteModalVisible(true);
      setSelectedConversationId(conversationId);
      setSelectedConversationTitle(title);
    },
    [],
  );

  const handleArchiveProject = React.useCallback(
    (conversationId: string, title: string) => {
      setConfirmArchiveModalVisible(true);
      setSelectedConversationId(conversationId);
      setSelectedConversationTitle(title);
    },
    [],
  );

  // Unarchiving needs no confirmation: it restores a row the user can archive
  // again in one click, and nothing about the conversation itself changes.
  const handleUnarchiveProject = React.useCallback(
    (conversationId: string) => {
      removeArchivedConversation(activeBackend.id, conversationId);
    },
    [activeBackend.id, removeArchivedConversation],
  );

  const handleStopConversation = React.useCallback((conversationId: string) => {
    setConfirmStopModalVisible(true);
    setSelectedConversationId(conversationId);
  }, []);

  const handleConversationTitleChange = React.useCallback(
    (conversationId: string, newTitle: string) => {
      updateConversation(
        { conversationId, newTitle },
        {
          onSuccess: () => {
            displaySuccessToast(t(I18nKey.CONVERSATION$TITLE_UPDATED));
          },
        },
      );
    },
    [t, updateConversation],
  );

  const handleConfirmDelete = () => {
    if (selectedConversationId) {
      const conversationId = selectedConversationId;
      deleteConversation(
        { conversationId },
        {
          onSuccess: () => {
            removeArchivedConversation(activeBackend.id, conversationId);
            if (conversationId === currentConversationId) {
              navigate("/conversations");
            }
          },
        },
      );
    }
  };

  const handleConfirmArchive = () => {
    if (!selectedConversationId) {
      return;
    }
    archiveConversation(activeBackend.id, selectedConversationId);
    unpinConversation(activeBackend.id, selectedConversationId);
    if (selectedConversationId === currentConversationId) {
      navigate("/conversations");
    }
  };

  const handleConfirmStop = () => {
    if (selectedConversationId) {
      pauseConversation({
        conversationId: selectedConversationId,
      });
    }
  };

  const handleConfirmDeleteAll = async () => {
    // Delete against the unfiltered loaded set so archived (currently hidden)
    // conversations are still removed from the server — matching the action's
    // "delete all conversations" label and confirmation count.
    const idsToDelete = allLoadedConversations.map((c) => c.id);
    const results = await Promise.allSettled(
      idsToDelete.map((conversationId) =>
        deleteConversationAsync({ conversationId }),
      ),
    );

    const deletedIds = results.flatMap((result, index) =>
      result.status === "fulfilled" ? [idsToDelete[index]] : [],
    );
    const failedCount = results.length - deletedIds.length;

    for (const conversationId of deletedIds) {
      removeArchivedConversation(activeBackend.id, conversationId);
    }

    if (
      currentConversationId !== null &&
      deletedIds.includes(currentConversationId)
    ) {
      navigate("/conversations");
    }

    if (failedCount > 0) {
      displayErrorToast(
        `${failedCount} conversation${failedCount === 1 ? "" : "s"} could not be deleted.`,
      );
    }
  };

  const renderConversationCard = React.useCallback(
    (
      conversation: (typeof conversations)[number],
      options?: { inPinnedSection?: boolean },
    ) => {
      const isPinned = pinnedIds.includes(conversation.id);
      const isArchived = archivedIdSet.has(conversation.id);
      if (compact) {
        return (
          <CompactConversationRow
            key={conversation.id}
            conversationId={conversation.id}
            title={conversation.title ?? ""}
            selectedRepository={{
              selected_repository: conversation.selected_repository,
              selected_branch: conversation.selected_branch,
              git_provider: conversation.git_provider as Provider,
            }}
            executionStatus={conversation.execution_status}
            sandboxStatus={conversation.sandbox_status}
            lastUpdatedAt={conversation.updated_at}
            createdAt={conversation.created_at}
            workspaceWorkingDir={
              conversation.selected_workspace ??
              conversation.workspace?.working_dir
            }
            isActive={conversation.id === currentConversationId}
            onClose={onClose}
            showRepositoryMetadata={showRepoBranchMetadata}
            llmModel={conversation.llm_model}
            showLlmProfiles={showLlmProfiles}
            agentKind={conversation.agent_kind}
            acpServer={conversation.acp_server}
            tags={conversation.tags}
            showTags={showTagsMetadata}
          />
        );
      }
      return (
        <Tooltip
          key={conversation.id}
          placement="right-start"
          delay={1000}
          closeDelay={100}
          isDisabled={
            !showHoverMetadata || openContextMenuId === conversation.id
          }
          disableAnimation={import.meta.env.MODE === "test"}
          className="max-w-none overflow-visible rounded-xl border border-[var(--oh-border)] bg-base-secondary p-0 text-white shadow-xl"
          content={
            <ConversationCardPreview
              title={conversation.title ?? ""}
              executionStatus={conversation.execution_status}
              sandboxStatus={conversation.sandbox_status}
              selectedRepository={{
                selected_repository: conversation.selected_repository,
                selected_branch: conversation.selected_branch,
                git_provider: conversation.git_provider as Provider,
              }}
              workspaceWorkingDir={
                conversation.selected_workspace ??
                conversation.workspace?.working_dir
              }
              llmModel={conversation.llm_model}
              agentKind={conversation.agent_kind}
              acpServer={conversation.acp_server}
              createdAt={conversation.created_at}
              tags={conversation.tags}
            />
          }
        >
          <NavigationLink
            to={backendScopedPath(`/conversations/${conversation.id}`)}
            onClick={onClose}
            className={cn(
              "block rounded-md transition-colors",
              openContextMenuId !== conversation.id &&
                "hover:bg-[var(--oh-surface)]",
              (conversation.id === currentConversationId ||
                openContextMenuId === conversation.id) &&
                "bg-[var(--oh-surface)]",
            )}
          >
            <ConversationCard
              onDelete={() =>
                handleDeleteProject(conversation.id, conversation.title ?? "")
              }
              // Exactly one direction is offered per row, so the menu always
              // reflects the conversation's current archived state.
              onArchive={
                isArchived
                  ? undefined
                  : () =>
                      handleArchiveProject(
                        conversation.id,
                        conversation.title ?? "",
                      )
              }
              onUnarchive={
                isArchived
                  ? () => handleUnarchiveProject(conversation.id)
                  : undefined
              }
              onStop={() => handleStopConversation(conversation.id)}
              onChangeTitle={(title) =>
                handleConversationTitleChange(conversation.id, title)
              }
              title={conversation.title ?? ""}
              selectedRepository={{
                selected_repository: conversation.selected_repository,
                selected_branch: conversation.selected_branch,
                git_provider: conversation.git_provider as Provider,
              }}
              lastUpdatedAt={conversation.updated_at}
              createdAt={conversation.created_at}
              executionStatus={conversation.execution_status}
              sandboxStatus={conversation.sandbox_status}
              conversationId={conversation.id}
              contextMenuOpen={openContextMenuId === conversation.id}
              onContextMenuToggle={(isOpen) =>
                setOpenContextMenuId(isOpen ? conversation.id : null)
              }
              isActive={conversation.id === currentConversationId}
              workspaceWorkingDir={
                conversation.selected_workspace ??
                conversation.workspace?.working_dir
              }
              showRepositoryMetadata={showRepoBranchMetadata}
              llmModel={conversation.llm_model}
              showLlmProfiles={showLlmProfiles}
              agentKind={conversation.agent_kind}
              acpServer={conversation.acp_server}
              tags={conversation.tags}
              showTags={showTagsMetadata}
              isArchived={isArchived}
              isPinned={isPinned}
              onTogglePin={() => togglePin(activeBackend.id, conversation.id)}
              alwaysShowPinIcon={isPinned && !options?.inPinnedSection}
            />
          </NavigationLink>
        </Tooltip>
      );
    },
    [
      activeBackend.id,
      archivedIdSet,
      compact,
      currentConversationId,
      handleArchiveProject,
      handleConversationTitleChange,
      handleDeleteProject,
      handleStopConversation,
      handleUnarchiveProject,
      onClose,
      openContextMenuId,
      pinnedIds,
      showRepoBranchMetadata,
      showLlmProfiles,
      showTagsMetadata,
      showHoverMetadata,
      togglePin,
    ],
  );

  // Standard layout: panel fills its slot in the sidebar; the inner scroll
  // child fills the panel and scrolls when its content overflows. Modals are
  // siblings of the scroll element and are `position: fixed`, so they don't
  // participate in the panel's scroll geometry.
  // Gate on `isLoading` / `!isFetched` (true only until the first fetch settles),
  // not `isFetching` — the latter flips back to true on every 10s background
  // refetch, causing the skeleton/empty-state to flicker when the list is empty.
  const showInitialSkeleton = isLoading || !isFetched;
  const showPinnedSection =
    !compact && !showInitialSkeleton && pinnedConversations.length > 0;
  const showEmptyState =
    isFetched &&
    !isLoading &&
    !compact &&
    listIsEffectivelyEmpty &&
    !showPinnedSection &&
    !startTasks?.length;

  const showConversationHeader = !compact;

  return (
    <div
      ref={ref}
      data-testid="conversation-panel"
      className="flex h-full min-h-0 w-full flex-col"
    >
      {showConversationHeader && (
        <div
          className={cn(
            // Pull flush to the sidebar edges: `-ml-2.5` matches aside `pl-2.5`;
            // width extends by that inset on the right now that aside is `pr-0`.
            "-ml-2.5 w-[calc(100%+0.625rem)] max-w-none box-border border-b",
            isListScrolled ? "border-[var(--oh-border)]" : "border-transparent",
          )}
        >
          <div
            data-testid="older-conversations-summary"
            className="flex min-w-0 flex-nowrap items-center gap-x-2 py-2 pl-4 pr-2.5 text-[var(--oh-muted)]"
          >
            <span className="min-w-0 truncate text-sm font-medium text-[var(--oh-muted)]">
              {t(I18nKey.SIDEBAR$CONVERSATIONS)}
            </span>
            <div className="ml-auto flex shrink-0 items-center gap-0.5">
              <ConversationPanelNewThreadPicker
                backendKind={activeBackend.kind}
              />
              <ConversationPanelFilterMenu
                filterMenuOpen={filterMenuOpen}
                setFilterMenuOpen={setFilterMenuOpen}
                menuRef={filterMenuRef}
                backendKind={activeBackend.kind}
                organizeMode={organizeMode}
                setOrganizeMode={setOrganizeMode}
                conversationSort={conversationSort}
                setConversationSort={setConversationSort}
                threadScope={threadScope}
                setThreadScope={setThreadScope}
                automationFilterMode={automationFilterMode}
                setAutomationFilterMode={setAutomationFilterMode}
                selectedAutomationNames={selectedAutomationNames}
                onToggleAutomationName={toggleAutomationName}
                automationNameFacets={automationNameFacets}
                showOlderConversations={showOlderConversations}
                showArchivedConversations={showArchivedConversations}
                toggleShowArchivedConversations={
                  toggleShowArchivedConversations
                }
                toggleShowOlderConversations={toggleShowOlderConversations}
                showRepoBranchMetadata={showRepoBranchMetadata}
                toggleShowRepoBranchMetadata={toggleShowRepoBranchMetadata}
                showLlmProfiles={showLlmProfiles}
                toggleShowLlmProfiles={toggleShowLlmProfiles}
                showTagsMetadata={showTagsMetadata}
                toggleShowTagsMetadata={toggleShowTagsMetadata}
                showHoverMetadata={showHoverMetadata}
                toggleShowHoverMetadata={toggleShowHoverMetadata}
                totalConversationsCount={allLoadedConversations.length}
                onRequestDeleteAll={() => setConfirmDeleteAllVisible(true)}
              />
            </div>
          </div>
        </div>
      )}

      <div
        ref={scrollContainerRef}
        data-testid="conversation-panel-list-scroll"
        onScroll={(event) => {
          setIsListScrolled(event.currentTarget.scrollTop > 0);
        }}
        className={cn(
          "flex min-h-0 flex-1 flex-col overflow-y-auto overflow-x-hidden overscroll-contain custom-scrollbar-always",
          !compact && "conversation-panel-list-scroll",
        )}
      >
        {showInitialSkeleton && <ConversationCardSkeleton compact={compact} />}

        {!compact && showEmptyState && (
          <div
            data-testid="conversation-panel-empty-state"
            className="flex min-h-0 flex-1 flex-col items-center justify-center px-4 py-8"
          >
            <p className="text-xs text-[var(--oh-muted)]">
              {t(
                emptyDueToAutomationFilter
                  ? I18nKey.CONVERSATION_PANEL$NO_AUTOMATION_MATCHES
                  : I18nKey.CONVERSATION$NO_CONVERSATIONS,
              )}
            </p>
          </div>
        )}

        {showPinnedSection ? (
          <ConversationPanelPinnedSection
            pinnedConversations={pinnedConversations}
            isPreviewExpanded={expandedPinnedPreview}
            onTogglePreviewExpanded={() =>
              setExpandedPinnedPreview((current) => !current)
            }
            activeConversationId={currentConversationId}
            showDivider={!compact && organizeMode === "chronological"}
            renderConversationCard={(conversation) =>
              renderConversationCard(conversation, { inPinnedSection: true })
            }
          />
        ) : null}

        {/* Render in-progress start tasks first (skipped in compact mode —
            their rich card layout doesn't fit in the icon rail). */}
        {!compact &&
          startTasks?.map((task) => (
            <NavigationLink
              key={task.id}
              to={backendScopedPath(`/conversations/task-${task.id}`)}
              onClick={onClose}
              className="block"
            >
              <StartTaskCard task={task} />
            </NavigationLink>
          ))}

        {!showInitialSkeleton && compact
          ? compactVisibleConversations.map((conversation) =>
              renderConversationCard(conversation),
            )
          : null}

        {!showInitialSkeleton &&
        !compact &&
        organizeMode === "grouped" &&
        orderedConversationGroups &&
        orderedConversationGroups.length > 0 ? (
          <ConversationGroupFolderList
            groups={orderedConversationGroups}
            groupIds={conversationGroupIds}
            groupFolderOrder={groupFolderOrder}
            setGroupFolderOrder={setGroupFolderOrder}
            collapsedGroupIds={collapsedGroupIds}
            expandedGroupPreviewIds={expandedGroupPreviewIds}
            discoveryConversationIds={groupDiscoveryConversationIds}
            onToggleGroupCollapsed={toggleGroupCollapsed}
            onToggleGroupPreviewExpanded={toggleGroupPreviewExpanded}
            isCreatingConversationFlow={isCreatingConversationFlow}
            activeConversationId={currentConversationId}
            onLaunchFromGroup={launchFromGroup}
            renderConversationCard={(conversation) =>
              renderConversationCard(conversation)
            }
          />
        ) : null}

        {!showInitialSkeleton &&
        !compact &&
        organizeMode === "chronological" ? (
          <div className="space-y-0.5">
            {sortedVisibleConversations.map((conversation) =>
              renderConversationCard(conversation),
            )}
          </div>
        ) : null}

        {/* Explicit "Load more" trigger. Only shown when more pages exist
            *and* the older list is currently visible (or there are no older
            conversations to begin with) — otherwise the next page would be
            populated mostly with conversations the user has chosen to hide. */}
        {showLoadMore &&
          (isLoadingMore ? (
            <div className="py-1">
              <ConversationCardSkeleton compact={compact} />
            </div>
          ) : (
            <div className="flex justify-center py-4">
              <button
                type="button"
                data-testid="load-more-conversations"
                onClick={requestLoadMore}
                className="text-xs text-[var(--oh-muted)] hover:text-white"
              >
                {t(I18nKey.CONVERSATION$LOAD_MORE)}
              </button>
            </div>
          ))}
      </div>

      {confirmDeleteModalVisible && (
        <ConfirmDeleteModal
          onConfirm={() => {
            handleConfirmDelete();
            setConfirmDeleteModalVisible(false);
            setSelectedConversationTitle(null);
          }}
          onCancel={() => {
            setConfirmDeleteModalVisible(false);
            setSelectedConversationTitle(null);
          }}
          conversationTitle={selectedConversationTitle ?? undefined}
        />
      )}

      {confirmArchiveModalVisible && (
        <ConfirmArchiveModal
          onConfirm={() => {
            handleConfirmArchive();
            setConfirmArchiveModalVisible(false);
            setSelectedConversationTitle(null);
          }}
          onCancel={() => {
            setConfirmArchiveModalVisible(false);
            setSelectedConversationTitle(null);
          }}
          conversationTitle={selectedConversationTitle ?? undefined}
        />
      )}

      {confirmDeleteAllVisible && (
        <ConfirmDeleteModal
          title={t(I18nKey.CONVERSATION$CONFIRM_DELETE_ALL_TITLE)}
          description={t(I18nKey.CONVERSATION$CONFIRM_DELETE_ALL_DESC, {
            count: allLoadedConversations.length,
          })}
          onConfirm={async () => {
            await handleConfirmDeleteAll();
            setConfirmDeleteAllVisible(false);
          }}
          onCancel={() => setConfirmDeleteAllVisible(false)}
        />
      )}

      {confirmStopModalVisible && (
        <ConfirmStopModal
          onConfirm={() => {
            handleConfirmStop();
            setConfirmStopModalVisible(false);
          }}
          onCancel={() => setConfirmStopModalVisible(false)}
        />
      )}

      {confirmExitConversationModalVisible && (
        <ExitConversationModal
          onConfirm={() => {
            onClose?.();
          }}
          onClose={() => setConfirmExitConversationModalVisible(false)}
          onCancel={() => setConfirmExitConversationModalVisible(false)}
        />
      )}
    </div>
  );
}
