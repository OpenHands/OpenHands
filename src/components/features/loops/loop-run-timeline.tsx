import { useTranslation } from "react-i18next";
import type { LoopRun, LoopRunStatus } from "#/api/loop-service/loop-types";
import { formatUsd } from "#/components/features/kanban/kanban-cost";
import { I18nKey } from "#/i18n/declaration";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import { cn } from "#/utils/utils";

const RUN_STATUS_KEY: Record<LoopRunStatus, I18nKey> = {
  pending: I18nKey.FEATURE_DEV$STATUS_PENDING,
  running: I18nKey.FEATURE_DEV$STATUS_RUNNING,
  passed: I18nKey.FEATURE_DEV$STATUS_PASSED,
  failed: I18nKey.FEATURE_DEV$STATUS_FAILED,
  aborted: I18nKey.FEATURE_DEV$STATUS_ABORTED,
  awaiting_input: I18nKey.FEATURE_DEV$STATUS_PAUSED,
};

const STAGE_STATUS_KEY: Record<string, I18nKey> = {
  pending: I18nKey.FEATURE_DEV$STATUS_PENDING,
  running: I18nKey.FEATURE_DEV$STATUS_RUNNING,
  passed: I18nKey.FEATURE_DEV$STATUS_PASSED,
  failed: I18nKey.FEATURE_DEV$STATUS_FAILED,
  skipped: I18nKey.LOOPS$STATUS_SKIPPED,
  aborted: I18nKey.FEATURE_DEV$STATUS_ABORTED,
};

const STATUS_CLASS: Record<string, string> = {
  passed: "text-green-400",
  failed: "text-red-400",
  running: "text-blue-400",
  aborted: "text-red-400",
  pending: "text-tertiary-light",
  awaiting_input: "text-amber-400",
};

export interface LoopRunTimelineProps {
  run: LoopRun;
  triggerReason?: string | null;
}

export function LoopRunTimeline({ run, triggerReason }: LoopRunTimelineProps) {
  const { t } = useTranslation("openhands");

  return (
    <section data-testid="loop-run-timeline" className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span
          data-testid="loop-run-status"
          className={cn(
            extensionModuleCardPillClassName,
            STATUS_CLASS[run.status] ?? "text-white",
          )}
        >
          {t(RUN_STATUS_KEY[run.status] ?? I18nKey.FEATURE_DEV$STATUS_PENDING)}
        </span>
        <span data-testid="loop-run-cost" className="tabular-nums text-white">
          {t(I18nKey.LOOPS$COST)}
          <span className="ml-2">{formatUsd(run.total_cost_usd)}</span>
        </span>
      </div>
      {triggerReason ? (
        <p
          data-testid="loop-run-reason"
          className="text-sm text-tertiary-light"
        >
          {t(I18nKey.LOOPS$TRIGGER_REASON)}
          <span className="ml-2 text-white">{triggerReason}</span>
        </p>
      ) : null}
      <ol className="flex flex-col gap-2">
        {run.stages.map((stage) => (
          <li
            key={stage.id}
            data-testid={`loop-stage-${stage.stage_name}`}
            className="rounded-xl bg-base-secondary p-3"
          >
            <div className="flex items-center justify-between gap-2">
              <p className="text-sm font-medium text-white">
                {stage.stage_name}
              </p>
              <span
                className={cn(
                  extensionModuleCardPillClassName,
                  STATUS_CLASS[stage.status] ?? "text-white",
                )}
              >
                {t(
                  STAGE_STATUS_KEY[stage.status] ??
                    I18nKey.FEATURE_DEV$STATUS_PENDING,
                )}
              </span>
            </div>
            {stage.last_output ? (
              <pre className="mt-2 overflow-x-auto text-xs text-tertiary-light">
                {stage.last_output}
              </pre>
            ) : null}
          </li>
        ))}
      </ol>
    </section>
  );
}
