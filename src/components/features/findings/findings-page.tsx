/**
 * Findings page shell — list, filter, triage.
 * @spec PROJETOSIN-188 — findings-page
 */

import React from "react";
import { useTranslation } from "react-i18next";
import {
  compareFindingsDefault,
  type Finding,
  type FindingSeverity,
  type FindingStatus,
} from "#/api/pentest/findings-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import {
  isFindingsForbiddenError,
  useFindingsList,
  useFindingsStats,
  useTriageFinding,
} from "#/hooks/query/use-findings";
import { useHasPentestCapability } from "#/hooks/use-pentest-capabilities";
import { I18nKey } from "#/i18n/declaration";
import { settingsLikeMainScrollClassName } from "#/utils/settings-like-page-layout-classes";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { FindingDetailDrawer } from "./finding-detail-drawer";
import { FindingFpModal } from "./finding-fp-modal";
import { FindingsDiffBanner } from "./findings-diff-banner";
import {
  EMPTY_FINDINGS_FILTERS,
  FindingsFilters,
  type FindingsFilterState,
} from "./findings-filters";
import {
  FindingsEmpty,
  FindingsEmptyNoEngagement,
  FindingsError,
  FindingsFilteredEmpty,
  FindingsForbidden,
  FindingsLoading,
} from "./findings-empty-state";
import {
  triageActionToStatus,
  type FindingsTriageAction,
} from "./findings-row-actions";
import { FindingsTable } from "./findings-table";

const PAGE_SIZE = 20;
const ASSET_DEBOUNCE_MS = 300;

export interface FindingsPageProps {
  engagementId: string | null;
  page: number;
  filters: FindingsFilterState;
  newOnly: boolean;
  onFiltersChange: (next: FindingsFilterState) => void;
  onClearFilters: () => void;
  onToggleNewOnly: () => void;
  onPageChange: (page: number) => void;
}

export function FindingsPage({
  engagementId,
  page,
  filters,
  newOnly,
  onFiltersChange,
  onClearFilters,
  onToggleNewOnly,
  onPageChange,
}: FindingsPageProps) {
  const { t, i18n } = useTranslation("openhands");
  const canView = useHasPentestCapability("pentest.findings.view");
  const canTriage = useHasPentestCapability("pentest.findings.triage");

  const [debouncedAsset, setDebouncedAsset] = React.useState(filters.asset);
  const [selectedFindingId, setSelectedFindingId] = React.useState<
    string | null
  >(null);
  const [fpTarget, setFpTarget] = React.useState<Finding | null>(null);
  const [fpError, setFpError] = React.useState<string | null>(null);

  React.useEffect(() => {
    const timer = window.setTimeout(() => {
      setDebouncedAsset(filters.asset.trim());
    }, ASSET_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [filters.asset]);

  const apiStatus = newOnly
    ? "new"
    : filters.statuses.length === 1
      ? filters.statuses[0]
      : undefined;
  const apiSeverity =
    filters.severities.length === 1 ? filters.severities[0] : undefined;
  const apiTool = filters.sourceTool.trim() || undefined;

  const listParams =
    engagementId && canView
      ? {
          engagement_id: engagementId,
          status: apiStatus,
          severity: apiSeverity,
          source_tool: apiTool,
          page,
          page_size: PAGE_SIZE,
        }
      : null;

  const listQuery = useFindingsList(listParams, {
    enabled: Boolean(engagementId) && canView,
  });
  const statsQuery = useFindingsStats(canView ? engagementId : null, {
    enabled: Boolean(engagementId) && canView,
  });
  const triageMutation = useTriageFinding();

  const hasActiveFilters =
    filters.severities.length > 0 ||
    filters.statuses.length > 0 ||
    Boolean(filters.sourceTool.trim()) ||
    Boolean(filters.asset.trim()) ||
    Boolean(filters.titleQuery.trim()) ||
    newOnly;

  const rawItems = listQuery.data?.items ?? [];
  const toolOptions = React.useMemo(() => {
    const tools = new Set<string>();
    for (const item of rawItems) {
      if (item.source_tool) tools.add(item.source_tool);
    }
    if (filters.sourceTool) tools.add(filters.sourceTool);
    return Array.from(tools).sort();
  }, [rawItems, filters.sourceTool]);

  const visibleFindings = React.useMemo(() => {
    let items = [...rawItems];

    if (filters.severities.length > 0) {
      const set = new Set<string>(filters.severities);
      items = items.filter((item) => set.has(item.severity));
    }
    if (filters.statuses.length > 0 && !newOnly) {
      const set = new Set<string>(filters.statuses);
      items = items.filter((item) => set.has(item.status));
    }
    if (debouncedAsset) {
      const needle = debouncedAsset.toLowerCase();
      items = items.filter((item) =>
        (item.asset ?? "").toLowerCase().includes(needle),
      );
    }
    if (filters.titleQuery.trim()) {
      const needle = filters.titleQuery.trim().toLowerCase();
      items = items.filter((item) => item.title.toLowerCase().includes(needle));
    }

    items.sort(compareFindingsDefault);
    return items;
  }, [
    rawItems,
    filters.severities,
    filters.statuses,
    filters.titleQuery,
    debouncedAsset,
    newOnly,
  ]);

  const runTriage = async (
    finding: Finding,
    action: FindingsTriageAction,
    fpReason?: string,
  ) => {
    try {
      await triageMutation.mutateAsync({
        findingId: finding.id,
        newStatus: triageActionToStatus(action),
        fpReason,
        currentStatus: finding.status,
      });
      displaySuccessToast(t(I18nKey.FINDINGS$TOAST_TRIAGE_SUCCESS));
      setFpTarget(null);
      setFpError(null);
    } catch {
      displayErrorToast(t(I18nKey.FINDINGS$TOAST_TRIAGE_ERROR));
      setFpError(t(I18nKey.FINDINGS$TOAST_TRIAGE_ERROR));
    }
  };

  const handleTriageAction = (
    finding: Finding,
    action: FindingsTriageAction,
  ) => {
    if (action === "mark_fp") {
      setFpError(null);
      setFpTarget(finding);
      return;
    }
    void runTriage(finding, action);
  };

  const forbiddenByApi =
    isFindingsForbiddenError(listQuery.error) ||
    isFindingsForbiddenError(statsQuery.error);

  const showListChrome =
    canView &&
    Boolean(engagementId) &&
    !forbiddenByApi &&
    !listQuery.isLoading &&
    !listQuery.isError;

  let content: React.ReactNode;
  if (!canView || forbiddenByApi) {
    content = <FindingsForbidden />;
  } else if (!engagementId) {
    content = <FindingsEmptyNoEngagement />;
  } else if (listQuery.isLoading || statsQuery.isLoading) {
    content = <FindingsLoading />;
  } else if (listQuery.isError) {
    content = <FindingsError onRetry={() => void listQuery.refetch()} />;
  } else if ((listQuery.data?.total ?? 0) === 0 && !hasActiveFilters) {
    content = <FindingsEmpty />;
  } else if (visibleFindings.length === 0) {
    content = <FindingsFilteredEmpty onClear={onClearFilters} />;
  } else {
    content = (
      <>
        <FindingsTable
          findings={visibleFindings}
          canTriage={canTriage}
          locale={i18n.language}
          onOpenDetail={(finding) => setSelectedFindingId(finding.id)}
          onTriageAction={handleTriageAction}
        />
        <FindingsPagination
          page={page}
          hasNext={Boolean(listQuery.data?.next_page)}
          onPageChange={onPageChange}
        />
      </>
    );
  }

  return (
    <main
      data-testid="findings-page"
      className={settingsLikeMainScrollClassName}
    >
      <header className="mb-6 flex flex-col gap-1">
        <h1 className="text-2xl font-semibold text-white">
          {t(I18nKey.FINDINGS$TITLE)}
        </h1>
        <p className="text-sm text-[var(--oh-text-secondary)]">
          {t(I18nKey.FINDINGS$SUBTITLE)}
        </p>
        {engagementId ? (
          <p className="mt-1 text-xs text-[var(--oh-text-tertiary)]">
            {t(I18nKey.FINDINGS$ENGAGEMENT_LABEL)}:{" "}
            <span className="font-mono">{engagementId}</span>
          </p>
        ) : (
          <p className="mt-1 text-xs text-[var(--oh-text-tertiary)]">
            {t(I18nKey.FINDINGS$ENGAGEMENT_MISSING_HINT)}
          </p>
        )}
      </header>

      <div className="flex flex-col gap-4">
        {showListChrome ? (
          <>
            <FindingsDiffBanner
              stats={statsQuery.data}
              newOnly={newOnly}
              onToggleNewOnly={onToggleNewOnly}
            />
            <FindingsFilters
              value={filters}
              toolOptions={toolOptions}
              onChange={onFiltersChange}
              onClear={onClearFilters}
              hasActiveFilters={hasActiveFilters}
            />
          </>
        ) : null}
        {content}
      </div>

      <FindingDetailDrawer
        findingId={selectedFindingId}
        canTriage={canTriage}
        locale={i18n.language}
        onClose={() => setSelectedFindingId(null)}
        onTriageAction={handleTriageAction}
      />

      <FindingFpModal
        isOpen={Boolean(fpTarget)}
        isPending={triageMutation.isPending}
        errorMessage={fpError}
        onCancel={() => {
          setFpTarget(null);
          setFpError(null);
        }}
        onSubmit={(reason) => {
          if (!fpTarget) return;
          void runTriage(fpTarget, "mark_fp", reason);
        }}
      />
    </main>
  );
}

function FindingsPagination({
  page,
  hasNext,
  onPageChange,
}: {
  page: number;
  hasNext: boolean;
  onPageChange: (page: number) => void;
}) {
  const { t } = useTranslation("openhands");
  return (
    <div className="flex items-center justify-between gap-3 pt-2">
      <BrandButton
        type="button"
        variant="secondary"
        isDisabled={page <= 1}
        onClick={() => onPageChange(page - 1)}
      >
        {t(I18nKey.FINDINGS$PAGINATION_PREV)}
      </BrandButton>
      <span className="text-xs text-[var(--oh-text-tertiary)]">
        {t(I18nKey.FINDINGS$PAGINATION_STATUS, { page })}
      </span>
      <BrandButton
        type="button"
        variant="secondary"
        isDisabled={!hasNext}
        onClick={() => onPageChange(page + 1)}
      >
        {t(I18nKey.FINDINGS$PAGINATION_NEXT)}
      </BrandButton>
    </div>
  );
}

export function parseSeveritiesParam(value: string | null): FindingSeverity[] {
  if (!value) return [];
  return value
    .split(",")
    .map((part) => part.trim())
    .filter((part): part is FindingSeverity =>
      ["critical", "high", "medium", "low", "info"].includes(part),
    );
}

export function parseStatusesParam(value: string | null): FindingStatus[] {
  if (!value) return [];
  return value
    .split(",")
    .map((part) => part.trim())
    .filter((part): part is FindingStatus =>
      [
        "new",
        "triaging",
        "confirmed",
        "false_positive",
        "duplicate",
        "risk_accepted",
      ].includes(part),
    );
}

export { EMPTY_FINDINGS_FILTERS };
