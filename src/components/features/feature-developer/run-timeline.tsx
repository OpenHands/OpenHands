import { useTranslation } from "react-i18next";
import type {
  FeatureDevRun,
  FeatureDevStatus,
  FeatureDevTicketStatus,
} from "#/api/feature-developer-service/feature-developer-types";
import type { GraphRelevantFile } from "#/api/graph-service/graph-types";
import { RelevantFilesCard } from "#/components/features/graph/relevant-files-card";
import { CostText } from "#/components/shared/cost-text";
import { I18nKey } from "#/i18n/declaration";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import { cn } from "#/utils/utils";

const RUN_STATUS_KEY: Record<FeatureDevStatus, I18nKey> = {
  pending: I18nKey.FEATURE_DEV$STATUS_PENDING,
  running: I18nKey.FEATURE_DEV$STATUS_RUNNING,
  passed: I18nKey.FEATURE_DEV$STATUS_PASSED,
  failed: I18nKey.FEATURE_DEV$STATUS_FAILED,
  partial: I18nKey.FEATURE_DEV$STATUS_PARTIAL,
  aborted: I18nKey.FEATURE_DEV$STATUS_ABORTED,
  paused: I18nKey.FEATURE_DEV$STATUS_PAUSED,
};

const TICKET_STATUS_KEY: Record<FeatureDevTicketStatus, I18nKey> = {
  pending: I18nKey.FEATURE_DEV$STATUS_PENDING,
  in_progress: I18nKey.FEATURE_DEV$STATUS_RUNNING,
  passed: I18nKey.FEATURE_DEV$STATUS_PASSED,
  failed: I18nKey.FEATURE_DEV$STATUS_FAILED,
  skipped: I18nKey.FEATURE_DEV$STATUS_ABORTED,
};

const STATUS_CLASS: Record<string, string> = {
  passed: "text-green-400",
  failed: "text-red-400",
  partial: "text-amber-400",
  running: "text-blue-400",
  in_progress: "text-blue-400",
  paused: "text-amber-400",
  aborted: "text-red-400",
  pending: "text-tertiary-light",
  skipped: "text-tertiary-light",
};

export interface RunTimelineProps {
  run: FeatureDevRun;
  graphFiles?: GraphRelevantFile[];
  graphEnabled?: boolean;
  graphDefaultEnabled?: boolean;
  graphBudgetLines?: number;
  graphUsedLines?: number;
  graphStaleSkipped?: boolean;
  onGraphToggle?: () => void;
}

export function RunTimeline({
  run,
  graphFiles,
  graphEnabled = true,
  graphDefaultEnabled = true,
  graphBudgetLines = 0,
  graphUsedLines = 0,
  graphStaleSkipped = false,
  onGraphToggle,
}: RunTimelineProps) {
  const { t } = useTranslation("openhands");

  return (
    <section data-testid="feature-dev-timeline" className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span
          data-testid="feature-dev-run-status"
          className={cn(
            extensionModuleCardPillClassName,
            STATUS_CLASS[run.status] ?? "text-white",
          )}
        >
          {t(RUN_STATUS_KEY[run.status] ?? I18nKey.FEATURE_DEV$STATUS_PENDING)}
        </span>
        <span
          data-testid="feature-dev-run-cost"
          className="tabular-nums text-white"
        >
          {t(I18nKey.FEATURE_DEV$TOTAL_COST)}
          <span className="ml-2">
            <CostText
              amount={run.total_actual_usd}
              at={run.updated_at ?? run.created_at}
            />
          </span>
        </span>
      </div>
      {onGraphToggle ? (
        <RelevantFilesCard
          files={graphFiles ?? []}
          budgetLines={graphBudgetLines}
          usedLines={graphUsedLines}
          enabled={graphEnabled}
          defaultEnabled={graphDefaultEnabled}
          staleSkipped={graphStaleSkipped}
          onToggle={onGraphToggle}
        />
      ) : null}
      <ol className="flex flex-col gap-2">
        {run.tickets.map((item) => (
          <li
            key={item.id}
            data-testid={`feature-dev-ticket-${item.id}`}
            className="rounded-xl bg-base-secondary p-3"
          >
            <div className="flex flex-wrap items-start justify-between gap-2">
              <p className="text-sm font-medium text-white">{item.title}</p>
              <span
                data-testid={`feature-dev-ticket-status-${item.id}`}
                className={cn(
                  extensionModuleCardPillClassName,
                  STATUS_CLASS[item.status] ?? "text-white",
                )}
              >
                {t(
                  TICKET_STATUS_KEY[item.status] ??
                    I18nKey.FEATURE_DEV$STATUS_PENDING,
                )}
              </span>
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-xs text-tertiary-light">
              <span data-testid={`feature-dev-ticket-cost-${item.id}`}>
                {t(I18nKey.FEATURE_DEV$COST)}
                <span className="ml-1 tabular-nums text-white">
                  <CostText
                    amount={item.actual_usd ?? item.estimate_usd}
                    at={item.finished_at ?? item.started_at ?? run.created_at}
                  />
                </span>
              </span>
              {item.branch_name ? (
                <span data-testid={`feature-dev-ticket-branch-${item.id}`}>
                  {t(I18nKey.FEATURE_DEV$BRANCH)}
                  <span className="ml-1 font-mono text-white">
                    {item.branch_name}
                  </span>
                </span>
              ) : null}
            </div>
            {item.error ? (
              <p
                data-testid={`feature-dev-ticket-error-${item.id}`}
                className="mt-2 text-sm text-red-400"
              >
                {item.error}
              </p>
            ) : null}
          </li>
        ))}
      </ol>
    </section>
  );
}
