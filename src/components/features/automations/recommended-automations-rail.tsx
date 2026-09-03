import {
  createElement,
  useCallback,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { useTranslation } from "react-i18next";
import type { RecommendedAutomation } from "@openhands/extensions/automations";
import {
  INTEGRATION_CATALOG as MCP_MARKETPLACE,
  type IntegrationCatalogEntry as MarketplaceEntry,
} from "@openhands/extensions/integrations";
import { I18nKey } from "#/i18n/declaration";
import { McpLogoBadge } from "#/components/features/mcp-logo-badge";
import { getMarketplaceEntryById } from "#/utils/mcp-marketplace-utils";
import {
  getAutomationIcon,
  getIntegrationIds,
} from "#/utils/automation-catalog";
import {
  flattenRecommendedRailGroups,
  getRecommendedRailGroups,
} from "#/utils/recommended-automation-rail";
import { readScrollFadeState } from "#/utils/scroll-fade-state";
import { useDragScroll } from "#/hooks/use-drag-scroll";
import type { Automation } from "#/types/automation";
import { AUTOMATION_STACK_SECTION_BOTTOM_CLASS } from "#/utils/automation-stack-section";
import { cn } from "#/utils/utils";
import {
  extensionModuleCardInteractiveClassName,
  extensionModuleCardSurfaceClassName,
} from "#/utils/extension-module-card-classes";

interface RecommendedAutomationsRailProps {
  installedAutomations: readonly Pick<Automation, "name">[];
  onSelect: (automation: RecommendedAutomation) => void;
  className?: string;
}

/** Current catalog tiles reserve 40px for the icon; keep that row height. */
const RAIL_ICON_ROW_CLASS_NAME = "flex h-10 items-start";

function integrationEntries(
  automation: RecommendedAutomation,
): MarketplaceEntry[] {
  return getIntegrationIds(automation).flatMap((id) => {
    const entry = getMarketplaceEntryById(id, MCP_MARKETPLACE);
    return entry ? [entry] : [];
  });
}

function RailIntegrationIcons({
  automation,
  entries,
  testId,
}: {
  automation: RecommendedAutomation;
  entries: MarketplaceEntry[];
  testId: string;
}) {
  const visibleEntries = entries.slice(0, 4);
  const isOverlap = visibleEntries.length > 1;
  // A declared glyph replaces the logos rather than joining them: it is the
  // whole identity of an entry that names one.
  const Icon = getAutomationIcon(automation);

  return (
    <span
      aria-hidden="true"
      data-testid={testId}
      data-layout={isOverlap && !Icon ? "overlap" : undefined}
      className={cn(
        RAIL_ICON_ROW_CLASS_NAME,
        isOverlap && !Icon && "-space-x-2",
      )}
    >
      {Icon ? (
        <McpLogoBadge
          entry={null}
          size="base"
          fallback={createElement(Icon, {
            className: "h-5 w-5",
            strokeWidth: 2.25,
          })}
        />
      ) : visibleEntries.length === 0 ? (
        <McpLogoBadge entry={null} size="base" />
      ) : (
        visibleEntries.map((entry) => (
          <McpLogoBadge
            key={entry.id}
            entry={entry}
            size="base"
            className={
              isOverlap
                ? "ring-2 ring-[var(--oh-color-base-secondary)]"
                : undefined
            }
          />
        ))
      )}
    </span>
  );
}

export function RecommendedAutomationsRail({
  installedAutomations,
  onSelect,
  className,
}: RecommendedAutomationsRailProps) {
  const { t } = useTranslation("openhands");
  const items = flattenRecommendedRailGroups(
    getRecommendedRailGroups(installedAutomations),
  );
  const scrollRef = useRef<HTMLDivElement>(null);
  const { handleMouseDown, handleClickCapture, handleDragStart } =
    useDragScroll(scrollRef);
  const [fadeState, setFadeState] = useState({ left: false, right: false });

  const updateFadeState = useCallback(() => {
    const element = scrollRef.current;
    if (!element) return;
    const next = readScrollFadeState(element);
    setFadeState((current) =>
      current.left === next.left && current.right === next.right
        ? current
        : next,
    );
  }, []);

  const itemIds = items.map((item) => item.id).join(",");

  useLayoutEffect(() => {
    updateFadeState();

    const element = scrollRef.current;
    if (!element) return undefined;

    const resizeObserver = new ResizeObserver(updateFadeState);
    resizeObserver.observe(element);
    Array.from(element.children).forEach((child) => {
      resizeObserver.observe(child);
    });

    return () => resizeObserver.disconnect();
  }, [updateFadeState, itemIds]);

  if (items.length === 0) return null;

  return (
    <section
      data-testid="recommended-automations-rail"
      aria-label={t(I18nKey.RECOMMENDED_AUTOMATIONS$SECTION_LABEL)}
      className={cn("w-full", AUTOMATION_STACK_SECTION_BOTTOM_CLASS, className)}
    >
      <h2 className="mb-2 text-sm font-medium text-[var(--oh-foreground)]">
        {t(I18nKey.RECOMMENDED_AUTOMATIONS$SECTION_LABEL)}
      </h2>

      <div className="relative">
        {/* eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions -- mouse drag-to-scroll is a pointer convenience; keyboard users scroll the list natively via the focusable cards. */}
        <div
          ref={scrollRef}
          role="list"
          data-testid="recommended-automations-rail-scroll"
          onScroll={updateFadeState}
          onMouseDown={handleMouseDown}
          onClickCapture={handleClickCapture}
          onDragStart={handleDragStart}
          className="flex flex-nowrap gap-3 overflow-x-auto scrollbar-hide select-none"
        >
          {items.map((automation) => (
            <div key={automation.id} role="listitem">
              <button
                type="button"
                data-testid={`recommended-automation-rail-card-${automation.id}`}
                onClick={() => onSelect(automation)}
                className={cn(
                  "flex w-[220px] shrink-0 flex-col gap-3 p-3 text-left",
                  extensionModuleCardSurfaceClassName,
                  extensionModuleCardInteractiveClassName,
                )}
              >
                <RailIntegrationIcons
                  automation={automation}
                  entries={integrationEntries(automation)}
                  testId={`recommended-automation-rail-icon-${automation.id}`}
                />
                <div className="min-w-0">
                  <h3 className="truncate text-sm font-semibold text-white">
                    {automation.name}
                  </h3>
                  <p className="mt-0.5 line-clamp-2 text-xs leading-relaxed text-tertiary-light">
                    {automation.description}
                  </p>
                </div>
              </button>
            </div>
          ))}
        </div>
        <div
          aria-hidden
          data-testid="recommended-automations-rail-fade-left"
          data-visible={fadeState.left ? "true" : "false"}
          className={cn(
            "pointer-events-none absolute inset-y-0 left-0 z-10 w-10",
            "bg-gradient-to-r from-[var(--oh-scroll-fade-from,var(--oh-color-base))] to-transparent",
            "transition-opacity duration-300 ease-out motion-reduce:transition-none",
            fadeState.left ? "opacity-100" : "opacity-0",
          )}
        />
        <div
          aria-hidden
          data-testid="recommended-automations-rail-fade-right"
          data-visible={fadeState.right ? "true" : "false"}
          className={cn(
            "pointer-events-none absolute inset-y-0 right-0 z-10 w-10",
            "bg-gradient-to-l from-[var(--oh-scroll-fade-from,var(--oh-color-base))] to-transparent",
            "transition-opacity duration-300 ease-out motion-reduce:transition-none",
            fadeState.right ? "opacity-100" : "opacity-0",
          )}
        />
      </div>
    </section>
  );
}
