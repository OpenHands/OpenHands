import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useAutomationRuns } from "#/hooks/query/use-automation-detail";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useTracking } from "#/hooks/use-tracking";
import ActivityIcon from "#/icons/activity.svg?react";
import { cn } from "#/utils/utils";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { downloadActivityLogExport } from "#/utils/automation-activity-log-export";
import type { ActivityLogExportFormat, Automation } from "#/types/automation";
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

export function ActivityLogSection({
  automation,
  highlightedRunId = null,
}: ActivityLogSectionProps) {
  const { t } = useTranslation("openhands");
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

  const hasMore = data ? data.total > data.runs.length : false;
  const canExport = !isLoading && (data?.total ?? 0) > 0 && !isExporting;

  const handleExport = async (format: ActivityLogExportFormat) => {
    if (!canExport) return;
    setIsExporting(true);
    try {
      await downloadActivityLogExport({
        automation,
        format,
        conversationBaseUrl: window.location.origin,
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
      className="rounded-2xl border border-[var(--oh-border)] bg-[var(--oh-surface)]"
    >
      <div className="flex items-center gap-2 border-b border-[var(--oh-border)] px-5 py-3">
        <span className="size-4 text-muted">
          <ActivityIcon className="size-4" />
        </span>
        <h3 className="text-sm font-medium text-content">
          {t(I18nKey.AUTOMATIONS$DETAIL$ACTIVITY_LOG)}
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
          {data.runs.map((run, index) => {
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
                className={cn(
                  index > 0 ? "border-t border-[var(--oh-border)]" : "",
                  isHighlighted && "bg-[var(--oh-focus)]/10",
                )}
              >
                <ActivityLogItem run={run} automation={automation} />
              </div>
            );
          })}

          {hasMore && (
            <div className="border-t border-[var(--oh-border)] px-5 py-3">
              <button
                type="button"
                onClick={() => setLimit((prev) => prev + PAGE_SIZE)}
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
