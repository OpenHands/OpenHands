import { Tooltip } from "@heroui/react";
import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";
import { AutomationRunStatus } from "#/types/automation";
import { KebabMenu } from "./kebab-menu";
import { useHasPermission } from "#/hooks/use-has-permission";
import { useNavigation } from "#/context/navigation-context";
import PlayIcon from "#/icons/play.svg?react";
import { SkillCardPillRow } from "#/components/features/skills/skill-card-pill-row";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { cn } from "#/utils/utils";
import { formatRelativeTime } from "#/utils/format-relative-time";
import { buildAutomationMetadataPills } from "./build-automation-pills";
import { buildAutomationMenuItems } from "./build-automation-menu-items";
import { automationIconActionButtonClassName } from "./automation-action-button-classes";
import { AutomationRunStats } from "./automation-run-insights";
import { automationCardStatusStripClassName } from "./automation-view-mode";
import {
  extensionModuleCardInteractiveClassName,
  extensionModuleCardSurfaceClassName,
} from "#/utils/extension-module-card-classes";
import { toLatestRunState } from "./to-latest-run-state";
import { RunPhase, shouldShowRunPhase } from "./detail/run-phase";
import { RunStatusBadge } from "./detail/run-status-badge";
import { AutomationRunActivitySparkline } from "#/components/features/home/featured-automations/automation-run-activity-sparkline";
import type { RunSummaryState } from "#/manifests/automation-insights";
import type { InterfaceListInsights } from "#/manifests/types";
import {
  getLastRunTimestamp,
  shortenAutomationErrorDetail,
  shouldShowAutomationErrorHovercard,
} from "#/components/features/home/featured-automations/automation-run-health";

/** Run insights shown when the manifest declares the dashboard surface. */
export interface AutomationInsightsProps {
  spec: InterfaceListInsights;
  state: RunSummaryState | undefined;
}

interface AutomationCardProps {
  automation: Automation;
  onToggle: (id: string, enabled: boolean) => void;
  onRunNow: (id: string) => void;
  isRunPending?: boolean;
  onDelete: (id: string) => void;
  onExport: (automation: Automation) => void;
  onEdit?: (id: string) => void;
  insights?: AutomationInsightsProps;
}

export function AutomationCard({
  automation,
  onToggle,
  onRunNow,
  isRunPending = false,
  onDelete,
  onExport,
  onEdit,
  insights,
}: AutomationCardProps) {
  const { navigate } = useNavigation();
  const { t, i18n } = useTranslation("openhands");
  const canManage = useHasPermission("manage_automations");

  const scheduleLabel =
    automation.trigger.schedule_human || automation.trigger.type;
  const pills = useMemo(
    () => buildAutomationMetadataPills(automation, scheduleLabel),
    [automation, scheduleLabel],
  );

  const handleView = () => {
    navigate?.(`/automations/${automation.id}`);
  };

  const menuItems = buildAutomationMenuItems({
    automation,
    t,
    canManage,
    onRunNow,
    isRunPending,
    onView: handleView,
    onExport,
    onEdit,
    onToggle,
    onDelete,
  });

  const runState = toLatestRunState(insights?.state);
  const { latestRun, recentRuns, isLoading, isError } = runState;
  const timestamp = latestRun ? getLastRunTimestamp(latestRun) : null;
  const errorDetail =
    latestRun?.status === AutomationRunStatus.FAILED
      ? latestRun.error_detail?.trim() || null
      : null;
  const shortErrorDetail = errorDetail
    ? shortenAutomationErrorDetail(errorDetail)
    : null;
  const showErrorHovercard =
    errorDetail != null &&
    shortErrorDetail != null &&
    shouldShowAutomationErrorHovercard(errorDetail, shortErrorDetail);
  const showPhase = shouldShowRunPhase(latestRun?.status);
  const disableAnimation = import.meta.env.MODE === "test";

  return (
    <div
      role="link"
      tabIndex={0}
      data-testid={`automation-card-${automation.id}`}
      onClick={handleView}
      onKeyDown={(event) => {
        if (event.key === "Enter") handleView();
      }}
      className={cn(
        "group relative flex min-w-0 flex-col overflow-hidden p-4 text-left",
        extensionModuleCardSurfaceClassName,
        extensionModuleCardInteractiveClassName,
      )}
    >
      <header className="flex flex-col gap-1.5">
        <div className="flex h-8 items-center justify-between gap-3">
          <h3 className="min-w-0 flex-1 truncate text-sm font-semibold leading-none text-[var(--oh-foreground)]">
            {automation.name}
          </h3>
          <div className="flex shrink-0 items-center gap-0.5">
            {canManage ? (
              <StyledTooltip
                content={t(I18nKey.AUTOMATIONS$RUN_NOW)}
                placement="top"
              >
                <button
                  type="button"
                  data-testid={`automation-run-now-${automation.id}`}
                  aria-label={t(I18nKey.AUTOMATIONS$RUN_NOW)}
                  aria-busy={isRunPending}
                  disabled={isRunPending || !automation.enabled}
                  onClick={(event) => {
                    event.stopPropagation();
                    onRunNow(automation.id);
                  }}
                  className={automationIconActionButtonClassName}
                >
                  <PlayIcon className="size-4 shrink-0" aria-hidden />
                </button>
              </StyledTooltip>
            ) : null}
            <KebabMenu
              items={menuItems}
              triggerClassName="opacity-70 group-hover:opacity-100"
            />
          </div>
        </div>
        {automation.prompt ? (
          <p className="line-clamp-2 text-xs leading-relaxed text-[var(--oh-text-secondary)]">
            {automation.prompt}
          </p>
        ) : null}
      </header>

      {pills.length > 0 || recentRuns.length > 0 ? (
        <div
          className={cn(
            "mt-3 flex items-center gap-3",
            pills.length > 0 ? "justify-between" : "justify-end",
          )}
        >
          {pills.length > 0 ? (
            <div className="min-w-0 flex-1 overflow-hidden">
              <SkillCardPillRow
                pills={pills}
                testId={`automation-pills-${automation.id}`}
              />
            </div>
          ) : null}
          {recentRuns.length > 0 ? (
            <AutomationRunActivitySparkline
              automationId={automation.id}
              runs={recentRuns}
              testId={`automation-activity-${automation.id}`}
            />
          ) : null}
        </div>
      ) : null}

      {insights ? (
        <div className={automationCardStatusStripClassName}>
          <div className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden">
            {isLoading ? (
              <div
                className="h-3.5 w-3/4 animate-pulse rounded bg-surface-raised motion-reduce:animate-none"
                aria-hidden="true"
              />
            ) : null}

            {!isLoading && isError ? (
              <p className="truncate text-[var(--oh-text-secondary)]">
                {t(I18nKey.FEATURED_AUTOMATIONS$STATUS_UNAVAILABLE)}
              </p>
            ) : null}

            {!isLoading && !isError && !latestRun ? (
              <p className="truncate text-[var(--oh-text-secondary)]">
                {t(I18nKey.AUTOMATIONS$DETAIL$NO_RUNS)}
              </p>
            ) : null}

            {latestRun ? (
              <>
                <RunStatusBadge status={latestRun.status} iconOnly showLabel />

                {showPhase ? (
                  <RunPhase
                    status={latestRun.status}
                    code={latestRun.phase_code}
                    label={latestRun.phase_label}
                    updatedAt={latestRun.phase_updated_at}
                  />
                ) : null}

                {shortErrorDetail ? (
                  showErrorHovercard && errorDetail ? (
                    <Tooltip
                      content={
                        <p className="max-w-xs whitespace-pre-wrap break-words p-2 text-xs">
                          {errorDetail}
                        </p>
                      }
                      placement="top"
                      closeDelay={100}
                      disableAnimation={disableAnimation}
                      className="rounded-xl border border-[var(--oh-border)] bg-base-secondary p-0 text-white shadow-xl"
                    >
                      <span className="min-w-0 flex-1 cursor-default truncate text-[var(--oh-status-error)]">
                        {shortErrorDetail}
                      </span>
                    </Tooltip>
                  ) : (
                    <p className="min-w-0 flex-1 truncate text-[var(--oh-status-error)]">
                      {shortErrorDetail}
                    </p>
                  )
                ) : null}
              </>
            ) : null}
          </div>

          {timestamp ? (
            <span
              data-testid={`automation-last-run-${automation.id}`}
              className="shrink-0 text-[var(--oh-text-secondary)]"
            >
              {formatRelativeTime(timestamp, i18n.language, t)}
            </span>
          ) : null}
        </div>
      ) : null}

      {insights ? (
        <div className="mt-3">
          <AutomationRunStats
            state={insights.state}
            copy={insights.spec.stats}
          />
        </div>
      ) : null}
    </div>
  );
}
