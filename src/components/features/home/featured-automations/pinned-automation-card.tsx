import { Tooltip } from "@heroui/react";
import { ExternalLink } from "lucide-react";
import { useMemo, useRef } from "react";
import { useTranslation } from "react-i18next";
import { buildAutomationMetadataPills } from "#/components/features/automations/build-automation-pills";
import { EditAutomationModal } from "#/components/features/automations/detail/edit-automation-modal";
import {
  RunPhase,
  shouldShowRunPhase,
} from "#/components/features/automations/detail/run-phase";
import { RunStatusBadge } from "#/components/features/automations/detail/run-status-badge";
import { AutomationRunStats } from "#/components/features/automations/automation-run-insights";
import { toRunSummaryState } from "#/components/features/automations/to-latest-run-state";
import { TurnOffConfirmationModal } from "#/components/features/automations/turn-off-confirmation-modal";
import { getDashboardSpec } from "#/manifests/automation-interface";
import { SkillCardPillRow } from "#/components/features/skills/skill-card-pill-row";
import { NavigationLink } from "#/components/shared/navigation-link";
import type { LatestAutomationRunState } from "#/hooks/query/use-latest-automation-runs";
import { useHomeAutomationActions } from "#/hooks/use-home-automation-actions";
import { getDemoConversationTitle } from "#/fixtures/home-automations-demo";
import { useUserConversation } from "#/hooks/query/use-user-conversation";
import { I18nKey } from "#/i18n/declaration";
import { AutomationRunStatus, type Automation } from "#/types/automation";
import { formatRelativeTime } from "#/utils/format-relative-time";
import { cn } from "#/utils/utils";
import { automationCardStatusStripClassName } from "#/components/features/automations/automation-view-mode";
import {
  extensionModuleCardInteractiveClassName,
  extensionModuleCardSurfaceClassName,
} from "#/utils/extension-module-card-classes";
import { AutomationRunActivitySparkline } from "./automation-run-activity-sparkline";
import { buildPinnedAutomationMenuItems } from "./build-pinned-automation-menu-items";
import { HomeAutomationMenu } from "./home-automation-menu";
import {
  getLastRunTimestamp,
  shortenAutomationErrorDetail,
  shouldShowAutomationErrorHovercard,
} from "./automation-run-health";

interface PinnedAutomationCardProps {
  automation: Automation;
  runState: LatestAutomationRunState;
  onUnpin: (automationId: string) => void;
  onDragStart: (automationId: string) => void;
  onDragOver: (automationId: string, position: "before" | "after") => void;
  onDrop: (automationId: string) => void;
  onDragEnd: () => void;
  isDropTarget: boolean;
  dropPosition: "before" | "after" | null;
  isDragging: boolean;
}

/**
 * Home pinned card. Shares the Automations dashboard tile chrome (surface,
 * header, pills, status strip) and keeps pin-only extras: drag-to-reorder,
 * conversation title, and unpin.
 */
export function PinnedAutomationCard({
  automation,
  runState,
  onUnpin,
  onDragStart,
  onDragOver,
  onDrop,
  onDragEnd,
  isDropTarget,
  dropPosition,
  isDragging,
}: PinnedAutomationCardProps) {
  const { t, i18n } = useTranslation("openhands");
  const { latestRun, recentRuns, isLoading, isError } = runState;
  const actions = useHomeAutomationActions(automation, latestRun);
  const conversationId = latestRun?.conversation_id ?? null;
  const demoConversationTitle = conversationId
    ? getDemoConversationTitle(conversationId)
    : undefined;
  const { data: conversation } = useUserConversation(
    demoConversationTitle !== undefined ? null : conversationId,
  );
  const conversationTitle =
    demoConversationTitle !== undefined
      ? demoConversationTitle
      : (conversation?.title ?? null);

  const timestamp = latestRun ? getLastRunTimestamp(latestRun) : null;
  const isTerminal =
    latestRun?.status === AutomationRunStatus.COMPLETED ||
    latestRun?.status === AutomationRunStatus.FAILED;
  const scheduleLabel =
    automation.trigger.schedule_human || automation.trigger.type;
  const pills = useMemo(
    () => buildAutomationMetadataPills(automation, scheduleLabel),
    [automation, scheduleLabel],
  );
  const showPhase = shouldShowRunPhase(latestRun?.status);
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
  const disableAnimation = import.meta.env.MODE === "test";
  const cardRef = useRef<HTMLElement>(null);
  const insights = getDashboardSpec()?.insights;

  const menuItems = buildPinnedAutomationMenuItems({
    automation,
    t,
    canManage: actions.canManage,
    canEdit: actions.canEdit,
    isRunPending: actions.isRunPending,
    isCancelPending: actions.isCancelPending,
    canCancel: actions.canCancel,
    onRunNow: actions.runNow,
    onCancelRun: actions.cancelRun,
    onView: actions.viewDetails,
    onEdit: actions.canEdit ? actions.openEdit : undefined,
    onTurnOff: actions.requestTurnOff,
    onUnpin: () => onUnpin(automation.id),
  });

  return (
    <article
      ref={cardRef}
      role="listitem"
      data-testid={`pinned-automation-card-${automation.id}`}
      onDragOver={(event) => {
        event.preventDefault();
        const { dataTransfer } = event;
        dataTransfer.dropEffect = "move";
        const rect = event.currentTarget.getBoundingClientRect();
        // 2-col grid: prefer horizontal edge when the pointer is clearly
        // left/right of center; otherwise use vertical before/after.
        const xRatio = (event.clientX - rect.left) / rect.width;
        const yRatio = (event.clientY - rect.top) / rect.height;
        const useHorizontal = xRatio < 0.3 || xRatio > 0.7;
        const position = useHorizontal
          ? xRatio < 0.5
            ? "before"
            : "after"
          : yRatio < 0.5
            ? "before"
            : "after";
        onDragOver(automation.id, position);
      }}
      onDrop={(event) => {
        event.preventDefault();
        onDrop(automation.id);
      }}
      onDragEnd={onDragEnd}
      className={cn(
        "group relative flex min-w-0 flex-col overflow-hidden p-4 text-left",
        extensionModuleCardSurfaceClassName,
        extensionModuleCardInteractiveClassName,
        isDragging && "opacity-50",
        isDropTarget &&
          dropPosition === "before" &&
          "shadow-[inset_0_2px_0_0_var(--oh-focus)]",
        isDropTarget &&
          dropPosition === "after" &&
          "shadow-[inset_0_-2px_0_0_var(--oh-focus)]",
      )}
    >
      {/*
        Top padding + title row is the drag surface. Title is not an <a> so
        browsers don't steal the gesture; click still opens details. The
        menu is opted out via data-no-drag.
      */}
      <header className="flex flex-col gap-1.5">
        <div
          draggable
          data-testid={`pinned-automation-drag-${automation.id}`}
          aria-label={t(I18nKey.FEATURED_AUTOMATIONS$REORDER_HANDLE, {
            name: automation.name,
          })}
          className="-mx-4 -mt-4 flex cursor-grab items-center justify-between gap-3 px-4 pt-4 active:cursor-grabbing"
          onDragStart={(event) => {
            const target = event.target as HTMLElement | null;
            if (target?.closest("[data-no-drag]")) {
              event.preventDefault();
              return;
            }
            const { dataTransfer } = event;
            dataTransfer.effectAllowed = "move";
            dataTransfer.setData("text/plain", automation.id);
            if (cardRef.current) {
              dataTransfer.setDragImage(cardRef.current, 24, 24);
            }
            onDragStart(automation.id);
          }}
          onDragEnd={onDragEnd}
        >
          <span
            role="link"
            tabIndex={0}
            title={t(I18nKey.FEATURED_AUTOMATIONS$VIEW_DETAILS)}
            className="h-8 min-w-0 flex-1 cursor-pointer truncate text-sm font-semibold leading-8 text-[var(--oh-foreground)] hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--oh-focus)]"
            onClick={() => actions.viewDetails()}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                actions.viewDetails();
              }
            }}
          >
            {automation.name}
          </span>

          <div data-no-drag className="flex h-8 shrink-0 items-center">
            <HomeAutomationMenu
              testId={`pinned-automation-menu-${automation.id}`}
              panelTestId={`pinned-automation-menu-panel-${automation.id}`}
              ariaLabel={t(I18nKey.FEATURED_AUTOMATIONS$ROW_MENU_LABEL, {
                name: automation.name,
              })}
              triggerClassName="opacity-70 group-hover:opacity-100"
              items={menuItems}
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
                testId={`pinned-automation-pills-${automation.id}`}
              />
            </div>
          ) : null}
          {recentRuns.length > 0 ? (
            <AutomationRunActivitySparkline
              automationId={automation.id}
              runs={recentRuns}
              testId={`pinned-automation-activity-${automation.id}`}
            />
          ) : null}
        </div>
      ) : null}

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
              <RunStatusBadge
                status={latestRun.status}
                iconOnly
                showLabel={
                  latestRun.status === AutomationRunStatus.PENDING ||
                  // Running with no conversation title yet → same treatment as Pending.
                  (latestRun.status === AutomationRunStatus.RUNNING &&
                    !conversationTitle)
                }
              />

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

              {conversationId && conversationTitle ? (
                <NavigationLink
                  to={`/conversations/${conversationId}`}
                  aria-label={conversationTitle}
                  title={conversationTitle}
                  className="group/conversation inline-flex min-w-0 items-center gap-1 text-[var(--oh-foreground)] hover:text-[var(--oh-text-secondary)] focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--oh-focus)]"
                >
                  <span className="truncate">{conversationTitle}</span>
                  <ExternalLink
                    className="size-3 shrink-0 opacity-0 transition-opacity group-hover/conversation:opacity-100 group-focus-visible/conversation:opacity-100"
                    aria-hidden="true"
                  />
                </NavigationLink>
              ) : null}

              {!conversationId && isTerminal ? (
                <p className="min-w-0 truncate text-[var(--oh-text-secondary)]">
                  {t(I18nKey.AUTOMATIONS$DETAIL$NO_CONVERSATION)}
                </p>
              ) : null}
            </>
          ) : null}
        </div>

        {timestamp ? (
          <span className="shrink-0 text-[var(--oh-text-secondary)]">
            {formatRelativeTime(timestamp, i18n.language, t)}
          </span>
        ) : null}
      </div>

      {insights ? (
        <div className="mt-3">
          <AutomationRunStats
            state={toRunSummaryState(runState)}
            copy={insights.stats}
          />
        </div>
      ) : null}

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
    </article>
  );
}
