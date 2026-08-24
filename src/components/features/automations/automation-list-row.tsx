import { Tooltip } from "@heroui/react";
import { useTranslation } from "react-i18next";
import { Zap } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";
import { KebabMenu } from "./kebab-menu";
import { useHasPermission } from "#/hooks/use-has-permission";
import { useNavigation } from "#/context/navigation-context";
import { NavigationLink } from "#/components/shared/navigation-link";
import PlayIcon from "#/icons/play.svg?react";
import ClockIcon from "#/icons/clock.svg?react";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { cn } from "#/utils/utils";
import { formatRelativeTime } from "#/utils/format-relative-time";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import { automationIconActionButtonClassName } from "./automation-action-button-classes";
import { buildAutomationMenuItems } from "./build-automation-menu-items";
import { automationActivityRowClassName } from "./automation-view-mode";
import { RunStatusBadge } from "./detail/run-status-badge";
import { AutomationRunActivitySparkline } from "#/components/features/home/featured-automations/automation-run-activity-sparkline";
import { AutomationHealthIndicator } from "#/components/features/home/featured-automations/automation-health-indicator";
import {
  HomeAutomationRunTooltip,
  getRunStatusLabelKey,
} from "#/components/features/home/featured-automations/home-automation-run-tooltip";
import {
  deriveRunHealth,
  formatTriggerSourceLabel,
  getLastRunTimestamp,
  getTriggerEventLabel,
  getTriggerScheduleLabel,
  getTriggerSource,
} from "#/components/features/home/featured-automations/automation-run-health";
import type { AutomationInsightsProps } from "./automation-card";
import { toLatestRunState } from "./to-latest-run-state";

interface AutomationListRowProps {
  automation: Automation;
  onToggle: (id: string, enabled: boolean) => void;
  onRunNow: (id: string) => void;
  isRunPending?: boolean;
  onDelete: (id: string) => void;
  onExport: (automation: Automation) => void;
  onEdit?: (id: string) => void;
  insights?: AutomationInsightsProps;
}

export function AutomationListRow({
  automation,
  onToggle,
  onRunNow,
  isRunPending = false,
  onDelete,
  onExport,
  onEdit,
  insights,
}: AutomationListRowProps) {
  const { navigate } = useNavigation();
  const { t, i18n } = useTranslation("openhands");
  const canManage = useHasPermission("manage_automations");

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

  const isEventTrigger = automation.trigger.type === "event";
  const TriggerIcon = isEventTrigger ? Zap : ClockIcon;
  const triggerEventLabel = getTriggerEventLabel(automation);
  const triggerScheduleLabel = getTriggerScheduleLabel(automation);
  const triggerSource = getTriggerSource(automation);
  const hasTriggerMeta = Boolean(
    triggerEventLabel || triggerScheduleLabel || triggerSource,
  );

  const runState = toLatestRunState(insights?.state);
  const health = deriveRunHealth(runState);
  const latestRun = runState.latestRun;
  const lastRunAt = latestRun
    ? getLastRunTimestamp(latestRun)
    : automation.last_triggered_at;
  const whenLabel = lastRunAt
    ? formatRelativeTime(lastRunAt, i18n.language, t)
    : null;
  const hasMeta =
    hasTriggerMeta || Boolean(whenLabel) || Boolean(latestRun?.status);
  const detailHref = `/automations/${encodeURIComponent(automation.id)}`;
  const statusLabelKey = getRunStatusLabelKey(runState);
  const disableAnimation = import.meta.env.MODE === "test";

  return (
    <li
      data-testid={`automation-list-row-${automation.id}`}
      className={automationActivityRowClassName}
    >
      <Tooltip
        content={
          <HomeAutomationRunTooltip
            automation={automation}
            runState={runState}
          />
        }
        placement="top-start"
        closeDelay={100}
        disableAnimation={disableAnimation}
        className="rounded-xl border border-[var(--oh-border)] bg-base-secondary p-0 text-white shadow-xl"
      >
        <NavigationLink
          to={detailHref}
          aria-label={`${automation.name} ${t(statusLabelKey)}`}
          className="flex min-w-0 flex-1 items-center justify-between gap-3 px-3 py-2.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[var(--oh-focus)]"
        >
          <div className="grid min-w-0 flex-1 grid-cols-[auto_minmax(0,1fr)] gap-x-2">
            <span
              data-testid={`automation-health-${automation.id}`}
              className="inline-flex h-5 w-2.5 shrink-0 items-center justify-center"
            >
              <AutomationHealthIndicator health={health} />
            </span>
            <span className="truncate text-sm font-medium leading-5 text-[var(--oh-foreground)]">
              {automation.name}
            </span>
            {hasMeta ? (
              <span className="col-start-2 mt-0.5 flex min-w-0 items-center gap-1.5 text-xs leading-4 text-[var(--oh-text-secondary)]">
                <TriggerIcon className="size-3 shrink-0" aria-hidden="true" />
                {triggerEventLabel ? (
                  <span className="truncate">{triggerEventLabel}</span>
                ) : null}
                {triggerScheduleLabel ? (
                  <span className="truncate">{triggerScheduleLabel}</span>
                ) : null}
                {triggerSource ? (
                  <span
                    className={cn(
                      extensionModuleCardPillClassName,
                      "shrink-0 px-1.5 py-0 text-[var(--oh-text-secondary)]",
                    )}
                  >
                    {formatTriggerSourceLabel(triggerSource)}
                  </span>
                ) : null}
                {whenLabel ? (
                  <>
                    {hasTriggerMeta ? (
                      <span className="shrink-0" aria-hidden="true">
                        ·
                      </span>
                    ) : null}
                    <span
                      data-testid={`automation-last-run-${automation.id}`}
                      className="truncate"
                    >
                      {whenLabel}
                    </span>
                  </>
                ) : null}
                {latestRun ? (
                  <>
                    {hasTriggerMeta || whenLabel ? (
                      <span className="shrink-0" aria-hidden="true">
                        ·
                      </span>
                    ) : null}
                    <RunStatusBadge status={latestRun.status} compact />
                  </>
                ) : null}
              </span>
            ) : null}
            <span className="sr-only">{t(statusLabelKey)}</span>
          </div>
        </NavigationLink>
      </Tooltip>

      <div className="flex shrink-0 items-center gap-1.5 pr-1.5">
        {insights ? (
          <AutomationRunActivitySparkline
            automationId={automation.id}
            runs={runState.recentRuns}
            testId={`automation-activity-${automation.id}`}
          />
        ) : null}
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
    </li>
  );
}
