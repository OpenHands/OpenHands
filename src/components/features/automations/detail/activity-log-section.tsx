import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useAutomationRuns } from "#/hooks/query/use-automation-detail";
import { useAutomationRunSummaries } from "#/hooks/query/use-automation-run-summaries";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useTracking } from "#/hooks/use-tracking";
import ActivityIcon from "#/icons/activity.svg?react";
import { cn } from "#/utils/utils";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { downloadActivityLogExport } from "#/utils/automation-activity-log-export";
import { getAgentCanvasBaseUrl } from "#/utils/base-path";
import { getAutomationRunDisplay } from "#/utils/automation-run-display";
import {
  formatRelativeTime,
  isInvalidTimestamp,
} from "#/utils/format-relative-time";
import { getDashboardSpec } from "#/manifests/automation-interface";
import {
  AutomationRunStatus,
  type ActivityLogExportFormat,
  type Automation,
  type AutomationRun,
} from "#/types/automation";
import {
  averageDurationDisplay,
  runCountDisplay,
  successRateDisplay,
} from "#/components/features/automations/automation-run-insights";
import { AutomationRunActivitySparkline } from "#/components/features/home/featured-automations/automation-run-activity-sparkline";
import { ActivityLogItem } from "./activity-log-item";

interface ActivityLogSectionProps {
  automation: Automation;
  /** Optional run id from `?run=` to scroll/highlight in the log. */
  highlightedRunId?: string | null;
}

const PAGE_SIZE = 20;

/**
 * Upper bound on auto-loading while searching for a `?run=` deep link, so a
 * stale or bogus run id cannot page through the entire run history.
 */
const HIGHLIGHT_AUTO_LOAD_MAX_RUNS = 100;

/** The newest runs of a successful stretch that stay visible above its fold. */
const STRETCH_VISIBLE_RUNS = 5;

/** A fold hides at least this many runs; fewer are simply shown. */
const COLLAPSE_MIN_RUNS = 2;

type RunRow =
  | { kind: "run"; run: AutomationRun }
  | { kind: "collapsed"; id: string; runs: AutomationRun[] };

interface DayGroup {
  key: string;
  label: string;
  rows: RunRow[];
}

function isSuccessfulRun(run: AutomationRun): boolean {
  const { badgeStatus } = getAutomationRunDisplay(run);
  return (
    badgeStatus === AutomationRunStatus.COMPLETED || badgeStatus === "success"
  );
}

function runDate(run: AutomationRun): Date {
  // A pending run has no start time yet; it belongs to today.
  return isInvalidTimestamp(run.started_at)
    ? new Date()
    : new Date(run.started_at);
}

/**
 * Folds every stretch of successful runs after its newest few into one row,
 * so a script that succeeds every minute reads as a handful of lines, not
 * sixty. A stretch holding the deep-linked run, or one the user opened, stays
 * expanded.
 */
function foldSuccessfulRuns(
  runs: AutomationRun[],
  expanded: ReadonlySet<string>,
  highlightedRunId: string | null,
): RunRow[] {
  const rows: RunRow[] = [];
  let index = 0;
  while (index < runs.length) {
    let end = index;
    while (end < runs.length && isSuccessfulRun(runs[end])) end += 1;
    const stretch = runs.slice(index, end);
    const shown = stretch.slice(0, STRETCH_VISIBLE_RUNS);
    const folded = stretch.slice(STRETCH_VISIBLE_RUNS);
    const foldId = folded[0] ? `fold-${folded[0].id}` : "";
    const keepOpen =
      expanded.has(foldId) || folded.some((run) => run.id === highlightedRunId);
    if (folded.length >= COLLAPSE_MIN_RUNS && !keepOpen) {
      shown.forEach((run) => rows.push({ kind: "run", run }));
      rows.push({ kind: "collapsed", id: foldId, runs: folded });
    } else {
      stretch.forEach((run) => rows.push({ kind: "run", run }));
    }
    if (end === index) {
      rows.push({ kind: "run", run: runs[index] });
      end += 1;
    }
    index = end;
  }
  return rows;
}

function groupRunsByDay(
  runs: AutomationRun[],
  locale: string,
  expanded: ReadonlySet<string>,
  highlightedRunId: string | null,
  showAll: boolean,
): DayGroup[] {
  const currentYear = new Date().getFullYear();
  const days: { key: string; label: string; runs: AutomationRun[] }[] = [];
  runs.forEach((run) => {
    const date = runDate(run);
    const key = date.toDateString();
    const last = days[days.length - 1];
    if (last?.key === key) {
      last.runs.push(run);
      return;
    }
    days.push({
      key,
      label: date.toLocaleDateString(locale, {
        weekday: "long",
        month: "long",
        day: "numeric",
        year: date.getFullYear() === currentYear ? undefined : "numeric",
      }),
      runs: [run],
    });
  });
  return days.map((day) => ({
    key: day.key,
    label: day.label,
    rows: showAll
      ? day.runs.map((run) => ({ kind: "run" as const, run }))
      : foldSuccessfulRuns(day.runs, expanded, highlightedRunId),
  }));
}

function RunStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="truncate text-xs text-muted">{label}</dt>
      <dd className="mt-0.5 truncate text-sm font-medium text-content">
        {value}
      </dd>
    </div>
  );
}

export function ActivityLogSection({
  automation,
  highlightedRunId = null,
}: ActivityLogSectionProps) {
  const { t, i18n } = useTranslation("openhands");
  const active = useActiveBackend();
  const { trackAutomationActivityLogExported } = useTracking();
  const [limit, setLimit] = useState(PAGE_SIZE);
  const highlightedRef = useRef<HTMLDivElement | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const { data, isLoading } = useAutomationRuns({
    id: automation.id,
    limit,
    offset: 0,
  });

  // The same 20-run sample the list card summarizes, so both show one number.
  const summaryAutomations = useMemo(() => [automation], [automation]);
  const summaryState = useAutomationRunSummaries(summaryAutomations).get(
    automation.id,
  );
  const statsCopy = getDashboardSpec()?.insights.stats ?? null;
  const recentRuns = summaryState?.summary?.recentRuns ?? [];
  const [expandedFolds, setExpandedFolds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  // Loading more asks to see older runs. Folding them would hide them inside
  // the stretch they extend, so once the user pages, every run shows under
  // its day.
  const [showAllRuns, setShowAllRuns] = useState(false);
  const dayGroups = useMemo(
    () =>
      groupRunsByDay(
        data?.runs ?? [],
        i18n.language,
        expandedFolds,
        highlightedRunId,
        showAllRuns,
      ),
    [data?.runs, i18n.language, expandedFolds, highlightedRunId, showAllRuns],
  );
  const isScript = !automation.prompt;

  const hasMore = data ? data.total > data.runs.length : false;
  const canExport = !isLoading && (data?.total ?? 0) > 0 && !isExporting;

  const handleExport = async (format: ActivityLogExportFormat) => {
    if (!canExport) return;
    setIsExporting(true);
    try {
      await downloadActivityLogExport({
        automation,
        format,
        conversationBaseUrl: getAgentCanvasBaseUrl(),
      });
      trackAutomationActivityLogExported({
        backendKind: active.backend.kind,
        format,
      });
    } catch (error) {
      displayErrorToast(
        getApiErrorMessage(
          error,
          t(I18nKey.AUTOMATIONS$DETAIL$EXPORT_ACTIVITY_LOG_ERROR),
        ),
      );
    } finally {
      setIsExporting(false);
    }
  };

  useEffect(() => {
    if (!highlightedRunId || !data?.runs.length) return;
    const index = data.runs.findIndex((run) => run.id === highlightedRunId);
    if (index < 0) {
      if (hasMore && limit < HIGHLIGHT_AUTO_LOAD_MAX_RUNS) {
        setLimit((prev) => prev + PAGE_SIZE);
      }
      return;
    }
    highlightedRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }, [data?.runs, hasMore, highlightedRunId, limit]);

  return (
    <div
      data-testid="automation-activity-log"
      className="rounded-2xl border border-border bg-surface"
    >
      <div className="flex items-center gap-2 border-b border-border px-5 py-3">
        <span className="size-4 text-muted">
          <ActivityIcon className="size-4" />
        </span>
        <h3 className="text-sm font-medium text-content">
          {t(I18nKey.AUTOMATIONS$DETAIL$RUNS)}
        </h3>
        <div className="ml-auto flex items-center gap-3">
          <button
            type="button"
            data-testid="activity-log-export-json"
            disabled={!canExport}
            onClick={() => void handleExport("json")}
            className="text-sm text-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
          >
            {t(I18nKey.AUTOMATIONS$DETAIL$EXPORT_JSON)}
          </button>
          <button
            type="button"
            data-testid="activity-log-export-csv"
            disabled={!canExport}
            onClick={() => void handleExport("csv")}
            className="text-sm text-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
          >
            {t(I18nKey.AUTOMATIONS$DETAIL$EXPORT_CSV)}
          </button>
        </div>
      </div>

      <div
        data-testid="automation-run-summary"
        className="flex flex-wrap items-end justify-between gap-4 border-b border-border px-5 py-4"
      >
        <dl className="grid min-w-0 flex-1 grid-cols-2 gap-4 sm:grid-cols-4">
          {statsCopy ? (
            <>
              <RunStat
                label={statsCopy.runs}
                value={runCountDisplay(summaryState)}
              />
              <RunStat
                label={statsCopy.recentSuccess}
                value={successRateDisplay(summaryState)}
              />
              <RunStat
                label={statsCopy.averageDuration}
                value={averageDurationDisplay(summaryState)}
              />
            </>
          ) : null}
          <RunStat
            label={t(I18nKey.AUTOMATIONS$DETAIL$LAST_RUN)}
            value={
              automation.last_triggered_at
                ? formatRelativeTime(
                    automation.last_triggered_at,
                    i18n.language,
                    t,
                  )
                : t(I18nKey.AUTOMATIONS$DETAIL$TIME_NEVER)
            }
          />
        </dl>
        {recentRuns.length > 0 ? (
          <AutomationRunActivitySparkline
            automationId={automation.id}
            runs={recentRuns}
            testId="automation-detail-activity"
          />
        ) : null}
      </div>

      {isLoading && (
        <div className="space-y-1 p-5">
          {Array.from({ length: 3 }).map((_, i) => (
            <div
              key={`skeleton-${i}`}
              className="flex items-center justify-between py-3"
            >
              <div className="h-5 w-64 animate-pulse rounded bg-surface-raised" />
              <div className="h-6 w-24 animate-pulse rounded-full bg-surface-raised" />
            </div>
          ))}
        </div>
      )}

      {!isLoading && data?.runs.length === 0 && (
        <p className="px-5 py-8 text-center text-sm text-muted">
          {t(I18nKey.AUTOMATIONS$DETAIL$NO_RUNS)}
        </p>
      )}

      {!isLoading && data && data.runs.length > 0 && (
        <div>
          {isScript ? (
            <p
              data-testid="automation-script-runs-note"
              className="border-b border-border px-5 py-2.5 text-xs text-muted"
            >
              {t(I18nKey.AUTOMATIONS$DETAIL$SCRIPT_RUNS_NOTE)}
            </p>
          ) : null}
          {dayGroups.map((day) => (
            <div key={day.key} data-testid="automation-run-day">
              <h4 className="border-b border-border bg-black/10 px-5 py-2 text-xs font-medium text-muted">
                {day.label}
              </h4>
              {day.rows.map((row, index) => {
                const rowClassName = cn(index > 0 && "border-t border-border");
                if (row.kind === "collapsed") {
                  return (
                    <div key={row.id} className={rowClassName}>
                      <button
                        type="button"
                        data-testid="automation-run-fold"
                        onClick={() =>
                          setExpandedFolds((prev) => new Set([...prev, row.id]))
                        }
                        className="w-full px-5 py-2 text-left text-xs text-muted hover:bg-surface-raised hover:text-foreground"
                      >
                        {t(I18nKey.AUTOMATIONS$DETAIL$COLLAPSED_RUNS, {
                          count: row.runs.length,
                        })}
                      </button>
                    </div>
                  );
                }
                const { run } = row;
                const isHighlighted = run.id === highlightedRunId;
                return (
                  <div
                    key={run.id}
                    ref={isHighlighted ? highlightedRef : undefined}
                    data-testid={
                      isHighlighted
                        ? `automation-run-highlight-${run.id}`
                        : undefined
                    }
                    className={cn(rowClassName, isHighlighted && "bg-focus/10")}
                  >
                    <ActivityLogItem run={run} automation={automation} />
                  </div>
                );
              })}
            </div>
          ))}

          {hasMore && (
            <div className="border-t border-border px-5 py-3">
              <button
                type="button"
                data-testid="automation-runs-load-more"
                onClick={() => {
                  setShowAllRuns(true);
                  setLimit((prev) => prev + PAGE_SIZE);
                }}
                className="text-sm text-muted hover:text-foreground"
              >
                {t(I18nKey.AUTOMATIONS$DETAIL$LOAD_MORE_RUNS)}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
