import { useState, useMemo, useCallback, type ReactNode } from "react";
import { RefreshCw, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import { I18nKey } from "#/i18n/declaration";
import {
  displaySuccessToast,
  displaySuccessToastWithLink,
  displayErrorToast,
} from "#/utils/custom-toast-handlers";
import { getApiErrorMessage } from "#/utils/api-error-message";
import {
  useAutomationDrafts,
  useAutomations,
  useToggleAutomation,
  useDeleteAutomation,
  useDeleteAutomationDraft,
  useDispatchAutomation,
  useDispatchAutomationDraft,
  useImportAutomation,
} from "#/hooks/query/use-automations";
import { useAutomationHealth } from "#/hooks/query/use-automation-health";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import {
  setAutomationSetupDraft,
  type AutomationSetupKind,
} from "#/api/automation-setup-draft-store";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useNavigation } from "#/context/navigation-context";
import { SearchInput } from "#/components/features/automations/search-input";
import { AutomationGroup } from "#/components/features/automations/automation-group";
import { AutomationViewToggle } from "#/components/features/automations/automation-view-toggle";
import {
  automationActivityListClassName,
  automationActivityRowClassName,
  readStoredAutomationViewMode,
  writeStoredAutomationViewMode,
  type AutomationViewMode,
} from "#/components/features/automations/automation-view-mode";
import { AutomationCardSkeleton } from "#/components/features/automations/automation-card-skeleton";
import { EmptyState } from "#/components/features/automations/empty-state";
import { ErrorState } from "#/components/features/automations/error-state";
import { BackendNotConfigured } from "#/components/features/automations/backend-not-configured";
import { DeleteConfirmationModal } from "#/components/features/automations/delete-confirmation-modal";
import { EditAutomationModal } from "#/components/features/automations/detail/edit-automation-modal";
import { AddAutomationMenu } from "#/components/features/automations/add-automation-menu";
import { AddAutomationModal } from "#/components/features/automations/add-automation-modal";
import { ImportAutomationModal } from "#/components/features/automations/import-automation-modal";
import { RecommendedAutomationsLauncher } from "#/components/features/automations/recommended-automations-launcher";
import { BrandButton } from "#/components/features/settings/brand-button";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { useTracking } from "#/hooks/use-tracking";
import { useAutomationPermissions } from "#/hooks/use-automation-permissions";
import type { Automation, AutomationSpec } from "#/types/automation";
import {
  getAutomationExportFilename,
  parseAutomationFile,
  serializeAutomation,
} from "#/utils/automation-export";
import {
  automationDetailPath,
  getDashboardSpec,
  getInterfaceCopy,
  hasAutomationInterface,
} from "#/manifests/automation-interface";
import {
  applyDashboardView,
  computeOverviewTile,
  matchesAutomationSearch,
} from "#/manifests/automation-insights";
import { interpolateValues } from "#/manifests/manifest-template";
import type {
  AutomationDraftApiResponse,
  DashboardSortValue,
  DashboardStatusValue,
  DashboardTriggerValue,
} from "#/manifests/types";
import { useAutomationRunSummaries } from "#/hooks/query/use-automation-run-summaries";
import { useAutomationSubPageNav } from "#/components/features/automations/dashboard/use-automation-sub-page-nav";
import { AutomationsDashboardControls } from "#/components/features/automations/dashboard/automations-dashboard-controls";
import { AutomationsFilteredEmptyState } from "#/components/features/automations/dashboard/automations-filtered-empty-state";
import { MANIFEST_ICON_BY_SLUG } from "#/components/features/manifest/manifest-icons";
import { ManifestOverviewTiles } from "#/components/features/manifest/manifest-overview-tiles";
import { ManifestSubpageLayout } from "#/components/features/manifest/manifest-subpage-layout";
import { cn, downloadBlob } from "#/utils/utils";
import { isDraftAutomation } from "#/utils/automation-state";
import { automationIconActionButtonClassName } from "#/components/features/automations/automation-action-button-classes";
import PlayIcon from "#/icons/play.svg?react";
import { buildAutomationDraftTags } from "#/utils/automation-draft-tags";
import { StatusBadge } from "#/components/features/automations/status-badge";

const PAGE_SIZE = 50;

/**
 * The page renders the interface manifest's copy, so without an admitted
 * manifest there is nothing to render: a 404, which the layout's error
 * boundary renders.
 */
export const clientLoader = () => {
  if (!hasAutomationInterface()) {
    throw new Response(null, { status: 404, statusText: "Not Found" });
  }
  return null;
};

function getDraftKind(draft: AutomationDraftApiResponse): AutomationSetupKind {
  if (draft.endpoint === "/v1") return "custom";
  if (draft.endpoint === "/v1/preset/plugin") return "plugin";
  return "prompt";
}

function getDraftTriggerSummary(
  draft: AutomationDraftApiResponse,
  t: TFunction,
) {
  const trigger = draft.draft.trigger as Record<string, unknown> | undefined;
  if (!trigger || typeof trigger !== "object") return null;
  if (trigger.type === "event") {
    const source = typeof trigger.source === "string" ? trigger.source : null;
    const event = Array.isArray(trigger.on)
      ? trigger.on
          .filter((item): item is string => typeof item === "string")
          .join(", ")
      : typeof trigger.on === "string"
        ? trigger.on
        : null;
    return (
      [source, event].filter(Boolean).join(" · ") ||
      t(I18nKey.AUTOMATIONS$DETAIL$TRIGGER_EVENT)
    );
  }
  const schedule =
    typeof trigger.schedule === "string" ? trigger.schedule : null;
  return schedule ?? t(I18nKey.AUTOMATIONS$DETAIL$TRIGGER_SCHEDULE);
}

function isEventTriggeredDraft(draft: AutomationDraftApiResponse): boolean {
  const trigger = draft.draft.trigger as Record<string, unknown> | undefined;
  return Boolean(
    trigger && typeof trigger === "object" && trigger.type === "event",
  );
}

function setupDraftFromServerDraft(draft: AutomationDraftApiResponse) {
  const body = draft.draft as Record<string, unknown>;
  const plugins = Array.isArray(body.plugins)
    ? body.plugins
        .map((plugin) =>
          plugin && typeof plugin === "object"
            ? (plugin as Record<string, unknown>).source
            : null,
        )
        .filter((source): source is string => typeof source === "string")
    : [];
  return {
    prompt: typeof body.prompt === "string" ? body.prompt : "",
    kind: getDraftKind(draft),
    ...(plugins.length > 0 ? { plugins } : {}),
  };
}

function matchesSavedDraftSearch(
  draft: AutomationDraftApiResponse,
  searchQuery: string,
): boolean {
  const normalized = searchQuery.trim().toLowerCase();
  if (!normalized) return true;
  return [
    draft.name ?? draft.id,
    String(draft.draft.prompt ?? ""),
    getDraftKind(draft),
    getDraftTriggerSummary(draft, ((key: string) => key) as TFunction) ?? "",
  ]
    .join("\n")
    .toLowerCase()
    .includes(normalized);
}

interface SavedDraftsGroupProps {
  drafts: AutomationDraftApiResponse[];
  onResume: (draft: AutomationDraftApiResponse) => void;
  onDelete: (draft: AutomationDraftApiResponse) => void;
  onTest: (draft: AutomationDraftApiResponse) => void;
  resumingDraftId: string | null;
  deletingDraftId: string | null;
  testingDraftId: string | null;
}

function SavedDraftsGroup({
  drafts,
  onResume,
  onDelete,
  onTest,
  resumingDraftId,
  deletingDraftId,
  testingDraftId,
}: SavedDraftsGroupProps) {
  const { t } = useTranslation("openhands");
  if (drafts.length === 0) return null;

  return (
    <section>
      <div className="flex items-center">
        <h2 className="text-base font-semibold text-foreground">
          {t(I18nKey.AUTOMATIONS$SAVED_DRAFTS)}
        </h2>
        <StatusBadge count={drafts.length} />
      </div>
      <ul className={cn(automationActivityListClassName, "mt-3")}>
        {drafts.map((draft) => {
          const title = draft.name ?? t(I18nKey.AUTOMATIONS$UNTITLED_DRAFT);
          const isResuming = resumingDraftId === draft.id;
          const isDeleting = deletingDraftId === draft.id;
          const isTesting = testingDraftId === draft.id;
          const canTestDirectly =
            draft.dispatchable && !isEventTriggeredDraft(draft);
          const isBusy = isResuming || isDeleting || isTesting;
          return (
            <li
              key={draft.id}
              data-testid={`automation-setup-draft-${draft.id}`}
              className={automationActivityRowClassName}
            >
              <button
                type="button"
                data-testid={`automation-setup-draft-resume-${draft.id}`}
                className="flex min-w-0 flex-1 items-center px-3 py-2.5 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus"
                disabled={isBusy}
                aria-label={
                  isResuming
                    ? t(I18nKey.AUTOMATION_SETUP$RESUMING_DRAFT)
                    : title
                }
                onClick={() => onResume(draft)}
              >
                <span
                  data-testid={`automation-setup-draft-open-${draft.id}`}
                  className="block min-w-0 truncate text-sm font-medium leading-5 text-foreground"
                >
                  {title}
                </span>
              </button>
              <div className="flex shrink-0 items-center gap-1.5 pr-1.5">
                {canTestDirectly && !isBusy ? (
                  <StyledTooltip
                    content={t(I18nKey.AUTOMATION_SETUP$TEST_RUN)}
                    placement="top"
                  >
                    <button
                      type="button"
                      data-testid={`automation-setup-draft-test-${draft.id}`}
                      aria-label={t(I18nKey.AUTOMATION_SETUP$TEST_DRAFT)}
                      onClick={() => onTest(draft)}
                      className={automationIconActionButtonClassName}
                    >
                      <PlayIcon className="size-4 shrink-0" aria-hidden />
                    </button>
                  </StyledTooltip>
                ) : (
                  <button
                    type="button"
                    data-testid={`automation-setup-draft-test-${draft.id}`}
                    aria-label={
                      isTesting
                        ? t(I18nKey.AUTOMATION_SETUP$STARTING_TEST)
                        : t(I18nKey.AUTOMATION_SETUP$TEST_DRAFT)
                    }
                    disabled
                    onClick={() => onTest(draft)}
                    className={automationIconActionButtonClassName}
                  >
                    <PlayIcon className="size-4 shrink-0" aria-hidden />
                  </button>
                )}
                <button
                  type="button"
                  data-testid={`automation-setup-draft-delete-${draft.id}`}
                  aria-label={t(I18nKey.AUTOMATION_SETUP$DELETE_DRAFT)}
                  disabled={isDeleting || isResuming || isTesting}
                  onClick={() => onDelete(draft)}
                  className={automationIconActionButtonClassName}
                >
                  <Trash2 className="size-4" aria-hidden />
                </button>
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

export default function AutomationsList() {
  const { t } = useTranslation("openhands");
  const interfaceCopy = getInterfaceCopy();
  // Admission is stable for the session; memo just keeps one identity.
  const dashboardSpec = useMemo(() => getDashboardSpec(), []);
  const subPageNav = useAutomationSubPageNav();
  // The manifest's dashboard surface is all-or-nothing, so either both are
  // present (dashboard mode) or neither is (today's plain list).
  const dashboard =
    dashboardSpec && subPageNav
      ? { spec: dashboardSpec, nav: subPageNav }
      : null;
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<DashboardStatusValue>("all");
  const [triggerFilter, setTriggerFilter] =
    useState<DashboardTriggerValue>("all");
  const [sortValue, setSortValue] = useState<DashboardSortValue>(
    dashboardSpec?.sort.default ?? "last-run",
  );
  const [viewMode, setViewMode] = useState<AutomationViewMode>(() =>
    readStoredAutomationViewMode(),
  );
  const [limit, setLimit] = useState(PAGE_SIZE);
  const [deleteTarget, setDeleteTarget] = useState<{
    id: string;
    name: string;
  } | null>(null);
  const [deleteDraftTarget, setDeleteDraftTarget] =
    useState<AutomationDraftApiResponse | null>(null);
  const [resumingDraftId, setResumingDraftId] = useState<string | null>(null);
  const [editTarget, setEditTarget] = useState<Automation | null>(null);
  const [isAddAutomationOpen, setIsAddAutomationOpen] = useState(false);
  const [importSpec, setImportSpec] = useState<AutomationSpec | null>(null);
  const [isImportOpen, setIsImportOpen] = useState(false);

  const active = useActiveBackend();
  const { navigate } = useNavigation();
  // Git Sync is org-level config, so its entry point requires
  // manage_automations (admins/owners) on every backend kind.
  const { canManage } = useAutomationPermissions();

  const {
    data: healthData,
    isLoading: isHealthLoading,
    refetch: refetchHealth,
  } = useAutomationHealth();

  const isBackendHealthy = healthData?.status === "ok";

  // Only fetch automations if the backend is healthy
  const { data, isLoading, isError, refetch } = useAutomations({
    limit,
    offset: 0,
    enabled: isBackendHealthy,
  });
  const {
    data: draftsData,
    isLoading: isDraftsLoading,
    isError: isDraftsError,
    refetch: refetchDrafts,
  } = useAutomationDrafts({
    limit,
    offset: 0,
    enabled: isBackendHealthy,
  });
  // One runs query per listed automation — dashboard mode only.
  const runSummaries = useAutomationRunSummaries(data?.automations ?? [], {
    enabled: isBackendHealthy && dashboard !== null,
  });
  const { trackPrebuiltAutomationEnabled, trackAutomationExported } =
    useTracking();
  const toggleMutation = useToggleAutomation();
  const deleteMutation = useDeleteAutomation();
  const deleteDraftMutation = useDeleteAutomationDraft();
  const dispatchMutation = useDispatchAutomation();
  const dispatchDraftMutation = useDispatchAutomationDraft();
  const importMutation = useImportAutomation();
  const createConversationMutation = useCreateConversation();

  const visible = useMemo(() => {
    if (!data?.automations) return [];
    if (!dashboardSpec) {
      return data.automations.filter((a) =>
        matchesAutomationSearch(a, searchQuery),
      );
    }
    return applyDashboardView(
      data.automations,
      {
        search: searchQuery,
        status: statusFilter,
        trigger: triggerFilter,
        sort: sortValue,
      },
      runSummaries,
    );
  }, [
    data?.automations,
    dashboardSpec,
    searchQuery,
    statusFilter,
    triggerFilter,
    sortValue,
    runSummaries,
  ]);

  const visibleSavedDrafts = useMemo(
    () =>
      (draftsData?.drafts ?? []).filter((draft) =>
        matchesSavedDraftSearch(draft, searchQuery),
      ),
    [draftsData?.drafts, searchQuery],
  );
  const activeAutomations = useMemo(
    () => visible.filter((a) => a.enabled && !isDraftAutomation(a)),
    [visible],
  );
  const inactive = useMemo(
    () => visible.filter((a) => !a.enabled && !isDraftAutomation(a)),
    [visible],
  );

  const handleToggle = (id: string, currentEnabled: boolean) => {
    const willEnable = !currentEnabled;
    toggleMutation.mutate({ id, enabled: willEnable });
    if (willEnable) {
      const automation = data?.automations.find((a) => a.id === id);
      trackPrebuiltAutomationEnabled({
        automationId: id,
        automationName: automation?.name ?? id,
      });
    }
  };

  const handleRunNow = (id: string) => {
    dispatchMutation.mutate(id, {
      onSuccess: () => {
        displaySuccessToast(t(I18nKey.AUTOMATIONS$RUN_NOW_SUCCESS));
      },
      onError: (error) => {
        displayErrorToast(
          getApiErrorMessage(error, t(I18nKey.AUTOMATIONS$RUN_NOW_ERROR)),
        );
      },
    });
  };

  const handleResumeDraft = (draft: AutomationDraftApiResponse) => {
    if (resumingDraftId) return;
    setResumingDraftId(draft.id);
    createConversationMutation.mutate(
      {
        query: t(I18nKey.AUTOMATIONS$CREATE_AUTOMATION_PROMPT),
        automationSetup: true,
        entryPoint: "automation_draft_resume",
      },
      {
        onSuccess: async (conversation) => {
          const conversationId = conversation.conversation_id;
          setAutomationSetupDraft(
            conversationId,
            setupDraftFromServerDraft(draft),
          );
          try {
            await AgentServerConversationService.updateConversationTags(
              conversationId,
              buildAutomationDraftTags(
                null,
                draft.id,
                draft.materializedAutomationId,
              ),
            );
          } catch {
            // Best-effort: the local setup draft still opens the form, and the
            // panel can save tags again after the next draft write.
          }
          navigate?.(`/conversations/${conversationId}`);
        },
        onError: (error) => {
          displayErrorToast(
            getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC)),
          );
        },
        onSettled: () => setResumingDraftId(null),
      },
    );
  };

  const handleTestDraft = (draft: AutomationDraftApiResponse) => {
    dispatchDraftMutation.mutate(draft.id, {
      onSuccess: () => {
        displaySuccessToast(t(I18nKey.AUTOMATION_SETUP$TEST_DISPATCHED));
      },
      onError: (error) => {
        displayErrorToast(
          getApiErrorMessage(error, t(I18nKey.AUTOMATIONS$RUN_NOW_ERROR)),
        );
      },
    });
  };

  const handleDeleteDraftRequest = (draft: AutomationDraftApiResponse) => {
    setDeleteDraftTarget(draft);
  };

  const handleDeleteRequest = (id: string) => {
    const automation = data?.automations.find((a) => a.id === id);
    if (automation) {
      setDeleteTarget({ id, name: automation.name });
    }
  };

  const handleEditRequest = (id: string) => {
    const automation = data?.automations.find((a) => a.id === id);
    if (automation) {
      setEditTarget(automation);
    }
  };

  const handleExport = (automation: Automation) => {
    const contents = `${JSON.stringify(serializeAutomation(automation), null, 2)}\n`;
    downloadBlob(
      new Blob([contents], { type: "application/json" }),
      getAutomationExportFilename(automation),
    );
    trackAutomationExported({ backendKind: active.backend.kind });
  };

  const handleImportFile = async (file: File) => {
    try {
      let parsed: unknown;
      try {
        parsed = JSON.parse(await file.text()) as unknown;
      } catch {
        displayErrorToast(t(I18nKey.AUTOMATIONS$IMPORT_INVALID_JSON));
        return;
      }
      setImportSpec(parseAutomationFile(parsed));
    } catch (error) {
      displayErrorToast(
        error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
      );
    }
  };

  const handleImportConfirm = () => {
    if (!importSpec) return;

    importMutation.mutate(
      { ...importSpec, enabled: false },
      {
        onSuccess: (created) => {
          setIsImportOpen(false);
          setImportSpec(null);
          displaySuccessToastWithLink(
            t(I18nKey.AUTOMATIONS$IMPORT_SUCCESS, { name: created.name }),
            t(I18nKey.AUTOMATIONS$IMPORT_VIEW),
            automationDetailPath(created.id),
          );
        },
        onError: (error) => {
          displayErrorToast(
            getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC)),
          );
        },
      },
    );
  };

  const handleDeleteConfirm = () => {
    if (deleteTarget) {
      deleteMutation.mutate(deleteTarget.id);
      setDeleteTarget(null);
    }
  };

  const handleDeleteDraftConfirm = () => {
    if (!deleteDraftTarget) return;
    deleteDraftMutation.mutate(deleteDraftTarget.id, {
      onSuccess: () => {
        displaySuccessToast(t(I18nKey.AUTOMATION_SETUP$DRAFT_DELETED));
      },
      onError: (error) => {
        displayErrorToast(getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC)));
      },
      onSettled: () => setDeleteDraftTarget(null),
    });
  };

  const handleViewModeChange = useCallback((view: AutomationViewMode) => {
    setViewMode(view);
    writeStoredAutomationViewMode(view);
  }, []);

  // Resets what filters to nothing: search and the dropdowns, never the sort.
  const handleClearFilters = () => {
    setSearchQuery("");
    setStatusFilter("all");
    setTriggerFilter("all");
  };

  const overviewTiles = useMemo(() => {
    if (!dashboardSpec) return [];
    const automations = data?.automations ?? [];
    return dashboardSpec.overview.tiles.map((tile) => {
      const value = computeOverviewTile(tile.metric, automations, runSummaries);
      const template =
        value.isZero && tile.zeroDetail ? tile.zeroDetail : tile.detail;
      return {
        key: tile.metric,
        label: tile.label,
        value: value.display,
        detail: interpolateValues(template, value.placeholderValues),
        Icon: MANIFEST_ICON_BY_SLUG[tile.icon],
      };
    });
  }, [dashboardSpec, data?.automations, runSummaries]);

  const groupInsights = dashboard
    ? { spec: dashboard.spec.insights, byId: runSummaries }
    : undefined;

  // Dashboard mode wraps the page in the manifest's sub-page shell; without a
  // manifest the wrapper — like everything else — is exactly today's.
  const renderShell = (content: ReactNode) =>
    dashboard ? (
      <ManifestSubpageLayout
        heading={dashboard.nav.heading}
        navTestIdBase="automations-navbar"
        items={dashboard.nav.items}
      >
        {content}
      </ManifestSubpageLayout>
    ) : (
      <div className="min-h-full">
        <div className="p-6 max-w-4xl mx-auto">{content}</div>
      </div>
    );

  const hasMore = data ? data.total > data.automations.length : false;
  const hasNoAutomations =
    !isLoading &&
    !isDraftsLoading &&
    !isError &&
    !isDraftsError &&
    data?.automations.length === 0 &&
    draftsData?.drafts.length === 0;

  // Show loading state while checking health
  if (isHealthLoading) {
    return renderShell(
      <div>
        <h1 className="text-xl font-medium text-content">
          {interfaceCopy.listTitle}
        </h1>
        <p className="mt-1 text-sm text-muted">{interfaceCopy.listSubtitle}</p>
        <div className="mt-6 flex flex-col gap-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <AutomationCardSkeleton key={`skeleton-${String(i)}`} />
          ))}
        </div>
      </div>,
    );
  }

  // Show backend not configured state if health check failed
  if (!isBackendHealthy) {
    return renderShell(
      <div>
        <h1 className="text-xl font-medium text-content">
          {interfaceCopy.listTitle}
        </h1>
        <p className="mt-1 text-sm text-muted">{interfaceCopy.listSubtitle}</p>
        <BackendNotConfigured onRetry={refetchHealth} />
      </div>,
    );
  }

  return renderShell(
    <>
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 flex-1 basis-64">
          <h1 className="text-xl font-semibold text-content">
            {interfaceCopy.listTitle}
          </h1>
          <p className="mt-1 text-sm text-muted">
            {interfaceCopy.listSubtitle}
          </p>
        </div>
        <div className="flex shrink-0 flex-wrap justify-end gap-2">
          {/* Git sync is org-level config, so it follows manage_automations
              (admins/owners) on every backend kind, not the local-only edit
              gate. */}
          {canManage && (
            <BrandButton
              type="button"
              variant="secondary"
              testId="automations-git-sync"
              className="whitespace-nowrap"
              onClick={() => navigate?.("/automations/git-sync")}
              startContent={<RefreshCw className="size-4" aria-hidden />}
            >
              {t(I18nKey.AUTOMATIONS$GIT_SYNC$NAV_BUTTON)}
            </BrandButton>
          )}
          <AddAutomationMenu
            onAdd={() => setIsAddAutomationOpen(true)}
            onImport={() => setIsImportOpen(true)}
          />
        </div>
      </div>

      {/* Overview tiles — dashboard mode only */}
      {dashboard && (
        <ManifestOverviewTiles
          label={dashboard.spec.overview.label}
          tiles={overviewTiles}
        />
      )}

      {/* Search */}
      <div
        className={cn(
          "flex items-stretch gap-2",
          dashboard ? "flex-wrap" : "mt-6",
        )}
      >
        <SearchInput value={searchQuery} onChange={setSearchQuery} />
        {dashboard && (
          <AutomationsDashboardControls
            spec={dashboard.spec}
            status={statusFilter}
            trigger={triggerFilter}
            sort={sortValue}
            onStatusChange={setStatusFilter}
            onTriggerChange={setTriggerFilter}
            onSortChange={setSortValue}
          />
        )}
        <AutomationViewToggle
          view={viewMode}
          onChange={handleViewModeChange}
          disabled={hasNoAutomations}
        />
      </div>

      {/* Content */}
      <div className={cn("flex flex-col gap-6", !dashboard && "mt-6")}>
        {(isLoading || isDraftsLoading) && (
          <div className="flex flex-col gap-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <AutomationCardSkeleton key={`skeleton-${String(i)}`} />
            ))}
          </div>
        )}

        {(isError || isDraftsError) && !(isLoading || isDraftsLoading) && (
          <ErrorState
            onRetry={() => {
              void refetch();
              void refetchDrafts();
            }}
          />
        )}

        {hasNoAutomations && <EmptyState />}

        {!isLoading &&
          !isError &&
          !isDraftsError &&
          data &&
          draftsData &&
          (data.automations.length > 0 || draftsData.drafts.length > 0) &&
          (dashboard &&
          visible.length === 0 &&
          visibleSavedDrafts.length === 0 ? (
            <AutomationsFilteredEmptyState onClear={handleClearFilters} />
          ) : (
            <>
              <SavedDraftsGroup
                drafts={visibleSavedDrafts}
                onResume={handleResumeDraft}
                onDelete={handleDeleteDraftRequest}
                onTest={handleTestDraft}
                resumingDraftId={resumingDraftId}
                deletingDraftId={
                  deleteDraftMutation.isPending
                    ? (deleteDraftMutation.variables ?? null)
                    : null
                }
                testingDraftId={
                  dispatchDraftMutation.isPending
                    ? (dispatchDraftMutation.variables ?? null)
                    : null
                }
              />
              <AutomationGroup
                title={t(I18nKey.AUTOMATIONS$ACTIVE)}
                count={activeAutomations.length}
                automations={activeAutomations}
                view={viewMode}
                onToggle={handleToggle}
                onRunNow={handleRunNow}
                runPendingId={
                  dispatchMutation.isPending
                    ? (dispatchMutation.variables ?? null)
                    : null
                }
                onDelete={handleDeleteRequest}
                onExport={handleExport}
                onEdit={handleEditRequest}
                insights={groupInsights}
              />
              <AutomationGroup
                title={t(I18nKey.AUTOMATIONS$INACTIVE)}
                count={inactive.length}
                automations={inactive}
                view={viewMode}
                onToggle={handleToggle}
                onRunNow={handleRunNow}
                runPendingId={
                  dispatchMutation.isPending
                    ? (dispatchMutation.variables ?? null)
                    : null
                }
                onDelete={handleDeleteRequest}
                onExport={handleExport}
                onEdit={handleEditRequest}
                insights={groupInsights}
              />

              {hasMore && (
                <button
                  type="button"
                  onClick={() => setLimit((prev) => prev + PAGE_SIZE)}
                  className="self-center rounded-lg border border-border px-6 py-2 text-sm text-contrast hover:bg-surface-raised"
                >
                  {t(I18nKey.AUTOMATIONS$LOAD_MORE)}
                </button>
              )}
            </>
          ))}
      </div>

      {/* The launcher lives on the templates sub-page in dashboard mode */}
      {!dashboard && (
        <div className="mt-6">
          <RecommendedAutomationsLauncher query={searchQuery} />
        </div>
      )}

      {/* Delete confirmation modal */}
      <DeleteConfirmationModal
        automationName={deleteTarget?.name ?? ""}
        isOpen={deleteTarget !== null}
        onConfirm={handleDeleteConfirm}
        onCancel={() => setDeleteTarget(null)}
      />

      {deleteDraftTarget ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <button
            type="button"
            className="absolute inset-0 bg-black/60"
            aria-label={t(I18nKey.BUTTON$CLOSE)}
            onClick={() => setDeleteDraftTarget(null)}
          />
          <div className="relative w-full max-w-sm rounded-xl border border-border bg-surface p-6">
            <h2 className="text-lg font-semibold text-content">
              {t(I18nKey.AUTOMATION_SETUP$DELETE_DRAFT_TITLE)}
            </h2>
            <p className="mt-2 text-sm text-muted">
              {t(I18nKey.AUTOMATION_SETUP$DELETE_DRAFT_MESSAGE, {
                name:
                  deleteDraftTarget.name ??
                  t(I18nKey.AUTOMATIONS$UNTITLED_DRAFT),
              })}
            </p>
            <div className="mt-6 flex justify-end gap-3">
              <button
                type="button"
                onClick={() => setDeleteDraftTarget(null)}
                className="rounded-lg border border-border px-4 py-2 text-sm text-contrast hover:bg-surface-raised"
              >
                {t(I18nKey.AUTOMATIONS$CANCEL)}
              </button>
              <button
                type="button"
                data-testid="automation-setup-draft-delete-confirm"
                onClick={handleDeleteDraftConfirm}
                disabled={deleteDraftMutation.isPending}
                className="rounded-lg bg-danger px-4 py-2 text-sm text-white hover:bg-danger/80 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {t(I18nKey.AUTOMATION_SETUP$DELETE_DRAFT)}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {/* Edit modal */}
      {editTarget && (
        <EditAutomationModal
          automation={editTarget}
          isOpen={editTarget !== null}
          onClose={() => setEditTarget(null)}
        />
      )}

      <AddAutomationModal
        isOpen={isAddAutomationOpen}
        onClose={() => setIsAddAutomationOpen(false)}
      />

      <ImportAutomationModal
        isOpen={isImportOpen}
        spec={importSpec}
        isImporting={importMutation.isPending}
        onClose={() => {
          setIsImportOpen(false);
          setImportSpec(null);
        }}
        onImport={handleImportConfirm}
        onFile={handleImportFile}
      />
    </>,
  );
}
