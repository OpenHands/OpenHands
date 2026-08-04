import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  UNKNOWN_RUN_STATE,
  useHomeAutomations,
} from "#/hooks/query/use-home-automations";
import {
  HOME_PINNED_PREVIEW_LIMIT,
  useHomePinnedAutomations,
} from "#/hooks/use-home-pinned-automations";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";
import { PinnedAutomationCard } from "./pinned-automation-card";

/**
 * Dashboard grid of pinned automations rendered above Recent Automation
 * Activity on the home page. Empty when nothing pinned resolves against
 * currently enabled live automations.
 */
export function PinnedAutomationsDashboard() {
  const { t } = useTranslation("openhands");
  const { pinnedIds, unpin, reorder, pruneMissing } =
    useHomePinnedAutomations();
  const {
    isBackendHealthy,
    isHealthLoading,
    isError,
    enabledAutomations,
    knownAutomationIds,
    isAutomationListComplete,
    runStates,
  } = useHomeAutomations();
  const [isExpanded, setIsExpanded] = useState(false);
  const [draggedId, setDraggedId] = useState<string | null>(null);
  const [dropTargetId, setDropTargetId] = useState<string | null>(null);
  const [dropPosition, setDropPosition] = useState<"before" | "after" | null>(
    null,
  );

  useEffect(() => {
    // Only prune against a complete id set: a truncated page would look like
    // missing automations and delete valid pins.
    if (!isAutomationListComplete || knownAutomationIds.size === 0) return;
    pruneMissing(knownAutomationIds);
  }, [isAutomationListComplete, knownAutomationIds, pruneMissing]);

  const enabledById = new Map(
    enabledAutomations.map((automation) => [automation.id, automation]),
  );
  const pinnedAutomations = pinnedIds
    .map((id) => enabledById.get(id))
    .filter((automation): automation is Automation => Boolean(automation));

  if (
    isHealthLoading ||
    !isBackendHealthy ||
    isError ||
    pinnedAutomations.length === 0
  ) {
    return null;
  }

  const visibleAutomations =
    isExpanded || pinnedAutomations.length <= HOME_PINNED_PREVIEW_LIMIT
      ? pinnedAutomations
      : pinnedAutomations.slice(0, HOME_PINNED_PREVIEW_LIMIT);
  const canExpand = pinnedAutomations.length > HOME_PINNED_PREVIEW_LIMIT;

  return (
    <section
      data-testid="pinned-automations-dashboard"
      aria-label={t(I18nKey.FEATURED_AUTOMATIONS$PINNED_DASHBOARD_LABEL)}
      className="w-full"
    >
      <h2 className="mb-2 text-sm font-medium text-[var(--oh-foreground)]">
        {t(I18nKey.FEATURED_AUTOMATIONS$PINNED_TITLE)}
      </h2>

      <div role="list" className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {visibleAutomations.map((automation) => (
          <PinnedAutomationCard
            key={automation.id}
            automation={automation}
            runState={runStates.get(automation.id) ?? UNKNOWN_RUN_STATE}
            onUnpin={unpin}
            onDragStart={setDraggedId}
            onDragOver={(id, position) => {
              setDropTargetId(id);
              setDropPosition(position);
            }}
            onDrop={(targetId) => {
              if (draggedId && dropPosition) {
                reorder(draggedId, targetId, dropPosition);
              }
              setDraggedId(null);
              setDropTargetId(null);
              setDropPosition(null);
            }}
            onDragEnd={() => {
              setDraggedId(null);
              setDropTargetId(null);
              setDropPosition(null);
            }}
            isDropTarget={dropTargetId === automation.id}
            dropPosition={dropTargetId === automation.id ? dropPosition : null}
            isDragging={draggedId === automation.id}
          />
        ))}
      </div>

      {canExpand ? (
        <div className="mt-2 flex justify-start">
          <button
            type="button"
            data-testid="pinned-automations-view-more"
            onClick={() => setIsExpanded((value) => !value)}
            className="text-xs text-[var(--oh-text-secondary)] hover:text-[var(--oh-foreground)]"
          >
            {t(
              isExpanded ? I18nKey.COMMON$VIEW_LESS : I18nKey.COMMON$VIEW_MORE,
            )}
          </button>
        </div>
      ) : null}
    </section>
  );
}
