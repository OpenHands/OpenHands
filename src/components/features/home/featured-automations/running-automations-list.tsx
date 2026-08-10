import { Tooltip } from "@heroui/react";
import { Plus, Zap } from "lucide-react";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { EditAutomationModal } from "#/components/features/automations/detail/edit-automation-modal";
import { RunStatusBadge } from "#/components/features/automations/detail/run-status-badge";
import { TurnOffConfirmationModal } from "#/components/features/automations/turn-off-confirmation-modal";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { NavigationLink } from "#/components/shared/navigation-link";
import type { LatestAutomationRunState } from "#/hooks/query/use-latest-automation-runs";
import {
  HOME_AUTOMATIONS_PREVIEW_LIMIT,
  MAX_HOME_AUTOMATION_CHIPS,
  UNKNOWN_RUN_STATE,
  useHomeAutomations,
} from "#/hooks/query/use-home-automations";
import { useHomeAutomationActions } from "#/hooks/use-home-automation-actions";
import { useHomePinnedAutomations } from "#/hooks/use-home-pinned-automations";
import { I18nKey } from "#/i18n/declaration";
import ClockIcon from "#/icons/clock.svg?react";
import type { Automation } from "#/types/automation";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import { cn } from "#/utils/utils";
import { AutomationRunActivitySparkline } from "./automation-run-activity-sparkline";
import { AutomationHealthIndicator } from "./automation-health-indicator";
import { buildHomeAutomationMenuItems } from "./build-home-automation-menu-items";
import { HomeAutomationMenu } from "./home-automation-menu";
import {
  deriveRunHealth,
  formatTriggerSourceLabel,
  getTriggerEventLabel,
  getTriggerScheduleLabel,
  getTriggerSource,
} from "./automation-run-health";
import {
  buildHomeAutomationActivityItems,
  hrefForActivityItem,
  type HomeAutomationActivityItem,
} from "./home-automation-activity";
import {
  getRunStatusLabelKey,
  HomeAutomationRunTooltip,
} from "./home-automation-run-tooltip";

interface RunningAutomationRowProps {
  item: HomeAutomationActivityItem;
  automation: Automation;
  runState: LatestAutomationRunState;
}

function RunningAutomationRow({
  item,
  automation,
  runState,
}: RunningAutomationRowProps) {
  const { t } = useTranslation("openhands");
  const { isPinned, togglePin } = useHomePinnedAutomations();
  const actions = useHomeAutomationActions(automation, runState.latestRun);
  const pinned = isPinned(item.id);
  const isEventTrigger = automation.trigger.type === "event";
  const TriggerIcon = isEventTrigger ? Zap : ClockIcon;
  const triggerEventLabel = getTriggerEventLabel(automation);
  const triggerScheduleLabel = getTriggerScheduleLabel(automation);
  const triggerSource = getTriggerSource(automation);
  const hasTriggerMeta = Boolean(
    triggerEventLabel || triggerScheduleLabel || triggerSource,
  );
  const hasMeta =
    hasTriggerMeta || Boolean(item.whenLabel) || Boolean(item.status);
  const health = deriveRunHealth(runState);
  const statusLabelKey = getRunStatusLabelKey(runState);
  const disableAnimation = import.meta.env.MODE === "test";

  const menuItems = buildHomeAutomationMenuItems({
    automation,
    t,
    canManage: actions.canManage,
    canEdit: actions.canEdit,
    isRunPending: actions.isRunPending,
    isCancelPending: actions.isCancelPending,
    canCancel: actions.canCancel,
    testIdPrefix: "running-automation",
    pinTestId: `running-automation-pin-${item.id}`,
    isPinned: pinned,
    onRunNow: actions.runNow,
    onCancelRun: actions.cancelRun,
    onView: actions.viewDetails,
    onEdit: actions.canEdit ? actions.openEdit : undefined,
    onTurnOff: actions.requestTurnOff,
    onTogglePin: () => togglePin(item.id),
  });

  return (
    <li
      data-testid={`running-automation-row-${item.id}`}
      className="group relative flex items-stretch transition-colors hover:bg-[var(--oh-interactive-hover)] has-[:focus-visible]:bg-[var(--oh-interactive-hover)]"
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
          to={hrefForActivityItem(item)}
          aria-label={`${item.name} ${t(statusLabelKey)}`}
          className="flex min-w-0 flex-1 items-center justify-between gap-3 px-3 py-2.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[var(--oh-focus)]"
        >
          <div className="grid min-w-0 flex-1 grid-cols-[auto_minmax(0,1fr)] gap-x-2">
            <span className="inline-flex h-5 w-2.5 shrink-0 items-center justify-center">
              <AutomationHealthIndicator health={health} />
            </span>
            <span className="truncate text-sm font-medium leading-5 text-[var(--oh-foreground)]">
              {item.name}
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
                {item.whenLabel ? (
                  <>
                    {hasTriggerMeta ? (
                      <span className="shrink-0" aria-hidden="true">
                        ·
                      </span>
                    ) : null}
                    <span className="truncate">{item.whenLabel}</span>
                  </>
                ) : null}
                {item.status ? (
                  <>
                    {hasTriggerMeta || item.whenLabel ? (
                      <span className="shrink-0" aria-hidden="true">
                        ·
                      </span>
                    ) : null}
                    <RunStatusBadge status={item.status} compact />
                  </>
                ) : null}
              </span>
            ) : null}
            <span className="sr-only">{t(statusLabelKey)}</span>
          </div>
        </NavigationLink>
      </Tooltip>

      <div className="flex shrink-0 items-center gap-1.5 pr-1.5">
        <AutomationRunActivitySparkline
          automationId={automation.id}
          runs={runState.recentRuns}
          testId={`running-automation-activity-${item.id}`}
        />
        <HomeAutomationMenu
          testId={`running-automation-menu-${item.id}`}
          panelTestId={`running-automation-menu-panel-${item.id}`}
          ariaLabel={t(I18nKey.FEATURED_AUTOMATIONS$ROW_MENU_LABEL, {
            name: item.name,
          })}
          triggerClassName="opacity-70 group-hover:opacity-100"
          items={menuItems}
        />
      </div>

      {actions.editOpen ? (
        <EditAutomationModal
          automation={automation}
          isOpen={actions.editOpen}
          onClose={() => actions.setEditOpen(false)}
        />
      ) : null}

      <TurnOffConfirmationModal
        automationName={automation.name}
        isOpen={actions.turnOffConfirmOpen}
        onConfirm={actions.confirmTurnOff}
        onCancel={actions.cancelTurnOff}
      />
    </li>
  );
}

/**
 * Recent automation activity under the home composer. Driven by live
 * enabled automations + latest-run queries; self-gates when the automation
 * service is unavailable or there are no enabled automations.
 */
export function RunningAutomationsList() {
  const { t, i18n } = useTranslation("openhands");
  const [isExpanded, setIsExpanded] = useState(false);
  const {
    isBackendHealthy,
    isHealthLoading,
    isError,
    isAutomationsLoading,
    enabledAutomations,
    runStates,
  } = useHomeAutomations();

  const language = i18n?.language ?? "en";
  const items = useMemo(
    () =>
      buildHomeAutomationActivityItems(
        enabledAutomations,
        runStates,
        language,
        t,
        UNKNOWN_RUN_STATE,
      ),
    [enabledAutomations, runStates, language, t],
  );

  const automationById = useMemo(
    () =>
      new Map(
        enabledAutomations.map((automation) => [automation.id, automation]),
      ),
    [enabledAutomations],
  );

  const visibleLimit = isExpanded
    ? MAX_HOME_AUTOMATION_CHIPS
    : HOME_AUTOMATIONS_PREVIEW_LIMIT;
  const visibleItems = items.slice(0, visibleLimit);
  const canViewMore =
    !isExpanded && items.length > HOME_AUTOMATIONS_PREVIEW_LIMIT;
  const canViewAll = isExpanded && items.length >= MAX_HOME_AUTOMATION_CHIPS;

  if (isHealthLoading || !isBackendHealthy || isError || isAutomationsLoading) {
    return null;
  }

  const isEmpty = items.length === 0;

  return (
    <section
      aria-labelledby="running-automations-heading"
      data-testid="running-automations-list"
      className="mt-8 w-full"
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <h2
          id="running-automations-heading"
          className="text-sm font-medium text-[var(--oh-foreground)]"
        >
          {t(I18nKey.FEATURED_AUTOMATIONS$SECTION_TITLE)}
        </h2>
        <StyledTooltip
          content={t(I18nKey.FEATURED_AUTOMATIONS$MANAGE)}
          placement="bottom"
        >
          <NavigationLink
            to="/automations"
            data-testid="home-automations-manage"
            aria-label={t(I18nKey.FEATURED_AUTOMATIONS$MANAGE)}
            className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs font-medium text-[var(--oh-text-secondary)] transition-colors hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--oh-foreground)] focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--oh-focus)]"
          >
            <Plus className="size-3" aria-hidden="true" />
            {t(I18nKey.BUTTON$ADD)}
          </NavigationLink>
        </StyledTooltip>
      </div>

      {isEmpty ? (
        <p
          data-testid="running-automations-empty-hint"
          className="rounded-xl border border-[var(--oh-border-subtle)] bg-[var(--oh-surface)] px-4 py-3 text-xs text-[var(--oh-text-secondary)]"
        >
          {t(I18nKey.FEATURED_AUTOMATIONS$EMPTY_HINT)}
        </p>
      ) : (
        <ul
          aria-label={t(I18nKey.FEATURED_AUTOMATIONS$RECENT_GROUP_LABEL)}
          className="divide-y divide-[var(--oh-border-subtle)] overflow-hidden rounded-xl border border-[var(--oh-border-subtle)] bg-[var(--oh-surface)]"
        >
          {visibleItems.map((item) => {
            const automation = automationById.get(item.id);
            if (!automation) return null;
            return (
              <RunningAutomationRow
                key={item.id}
                item={item}
                automation={automation}
                runState={runStates.get(item.id) ?? UNKNOWN_RUN_STATE}
              />
            );
          })}
        </ul>
      )}

      {canViewMore ? (
        <div className="mt-2 flex justify-start">
          <button
            type="button"
            data-testid="home-automations-view-more"
            onClick={() => setIsExpanded(true)}
            className="rounded-md px-1.5 py-0.5 text-xs font-medium text-[var(--oh-text-secondary)] transition-colors hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--oh-foreground)] focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--oh-focus)]"
          >
            {t(I18nKey.COMMON$VIEW_MORE)}
          </button>
        </div>
      ) : null}

      {canViewAll ? (
        <div className="mt-2 flex justify-start">
          <NavigationLink
            to="/automations"
            data-testid="home-automations-view-all"
            className="rounded-md px-1.5 py-0.5 text-xs font-medium text-[var(--oh-text-secondary)] transition-colors hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--oh-foreground)] focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--oh-focus)]"
          >
            {t(I18nKey.FEATURED_AUTOMATIONS$VIEW_ALL)}
          </NavigationLink>
        </div>
      ) : null}
    </section>
  );
}
