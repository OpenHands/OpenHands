import { createElement } from "react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import { I18nKey } from "#/i18n/declaration";
import {
  AUTOMATION_CATALOG,
  type RecommendedAutomation,
} from "@openhands/extensions/automations";
import {
  INTEGRATION_CATALOG as MCP_MARKETPLACE,
  type IntegrationCatalogEntry as MarketplaceEntry,
} from "@openhands/extensions/integrations";
import { McpLogoStackBadge } from "#/components/features/mcp-page/mcp-logo-stack-badge";
import { McpLogoBadge } from "#/components/features/mcp-logo-badge";
import {
  SkillCardPillRow,
  type SkillCardPill,
} from "#/components/features/skills/skill-card-pill-row";
import { CirclePlusBadge } from "#/components/shared/buttons/circle-plus-check-toggle";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { MCPServerConfig } from "#/types/mcp-server";
import type { NativeGitIntegration } from "#/hooks/query/use-native-git-integrations";
import {
  findInstalledEntryMatch,
  getMarketplaceEntryById,
  isMcpInstallableEntry,
} from "#/utils/mcp-marketplace-utils";
import { getFeaturedAutomationIds } from "#/manifests/automation-interface";
import {
  getAutomationIcon,
  getAutomationLaunchPrompt,
  getIntegrationIds,
  getTemplateIntegrationIds,
  getTemplateTriggerKinds,
  splitBySuggestion,
  type TemplateTriggerKind,
} from "#/utils/automation-catalog";
import { getAutomationsByPopularity } from "#/utils/recommended-automation-rail";
import { cn } from "#/utils/utils";
import {
  extensionModuleCardInteractiveClassName,
  extensionModuleCardGridClassName,
  extensionModuleCardGridContainerClassName,
  extensionModuleCardPillClassName,
  extensionModuleCardSurfaceClassName,
} from "#/utils/extension-module-card-classes";
import { AutomationsFilteredEmptyState } from "./dashboard/automations-filtered-empty-state";
import { StatusBadge } from "./status-badge";
import {
  EMPTY_TEMPLATE_FILTERS,
  hasActiveTemplateFilter,
  TemplateFilterChips,
  TRIGGER_KIND_ICON,
  TRIGGER_KIND_LABEL_KEY,
  type TemplateFilters,
} from "./template-filter-chips";

type GetNativeIntegration = (entryId: string) => NativeGitIntegration | null;

interface RecommendedAutomationsSectionProps {
  backendKind: "local" | "cloud";
  installedServers: MCPServerConfig[];
  /** Catalog entries the active backend's deployment does not offer. */
  hiddenAutomationIds?: ReadonlySet<string>;
  /** Entries the instance does not appear to support; listed last. */
  unlikelyAutomations?: ReadonlyMap<string, readonly string[]>;
  /** Entries using an integration the instance reports; suggested first. */
  suggestedAutomations?: ReadonlyMap<string, readonly string[]>;
  /** Resolves an integration's native (cloud) connection; none by default. */
  getNativeIntegration?: GetNativeIntegration;
  query?: string;
  /** Resets the search; with it, a no-match state offers a way back. */
  onClearQuery?: () => void;
  /** The integration and trigger filters, when the page offers them. */
  filters?: TemplateFilters;
  onFiltersChange?: (next: TemplateFilters) => void;
  onSelect: (automation: RecommendedAutomation) => void;
  /** When true, title, description, and cards share one scroll area. */
  scrollableGrid?: boolean;
}

export { getAutomationsByPopularity };

const RECOMMENDED_AUTOMATIONS = getAutomationsByPopularity(AUTOMATION_CATALOG);

// Proven automations are suggested on any backend; the rest carry a Beta tag.
// The set is owned by the interface manifest (falling back to the host
// default), not derived from popularityRank: slack-standup-digest@94 outranks
// slack-channel-monitor@92 yet is Beta.
function isProvenAutomation(automation: RecommendedAutomation): boolean {
  return getFeaturedAutomationIds().includes(automation.id);
}

export interface AutomationIntegration {
  id: string;
  entry?: MarketplaceEntry;
  /** False when this backend has no MCP install flow for the entry. */
  mcpInstallable: boolean;
}

/**
 * Every integration the automation declares, including the ones it is willing
 * to start without: the card describes what the automation uses, so an
 * optional integration still belongs on it. Entries are resolved against the
 * full catalog — an integration this backend cannot install as MCP (e.g.
 * Jira's HTTP-only option) is still a declared dependency and must stay
 * visible rather than being silently dropped. Unknown IDs are represented as
 * unresolved entries so catalog drift is visible to the user as well.
 */
function getIntegrationEntries(
  automation: RecommendedAutomation,
): AutomationIntegration[] {
  return getIntegrationIds(automation).map((id) => {
    const entry = getMarketplaceEntryById(id, MCP_MARKETPLACE);
    return {
      id,
      entry,
      mcpInstallable: !!entry && isMcpInstallableEntry(entry),
    };
  });
}

function automationMatchesFilters(
  automation: RecommendedAutomation,
  filters: TemplateFilters,
) {
  if (
    filters.integrationId &&
    !getTemplateIntegrationIds(automation).includes(filters.integrationId)
  ) {
    return false;
  }
  return (
    !filters.trigger ||
    getTemplateTriggerKinds(automation).includes(filters.trigger)
  );
}

/** The integrations the templates are about, in first-appearance order. */
function collectIntegrationIds(automations: RecommendedAutomation[]) {
  const ids: string[] = [];
  for (const automation of automations) {
    for (const id of getTemplateIntegrationIds(automation)) {
      if (!ids.includes(id)) ids.push(id);
    }
  }
  return ids;
}

function automationMatchesQuery(
  automation: RecommendedAutomation,
  integrations: AutomationIntegration[],
  rawQuery: string,
) {
  const query = rawQuery.trim().toLowerCase();
  if (!query) return true;
  const haystack = [
    automation.name,
    automation.category,
    automation.description,
    getAutomationLaunchPrompt(automation),
    ...integrations.flatMap(({ entry, id }) =>
      entry ? [entry.name, id, ...(entry.keywords ?? [])] : [id],
    ),
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(query);
}

function buildRecommendedAutomationPills(
  integrations: AutomationIntegration[],
  installedServers: MCPServerConfig[],
  getNativeIntegration: GetNativeIntegration,
  missingCount: number,
  translate: TFunction,
): SkillCardPill[] {
  const pills: SkillCardPill[] = integrations.map(
    ({ id, entry, mcpInstallable }) => {
      const native = getNativeIntegration(id);
      const installed =
        !!native?.isConnected ||
        (!!entry && findInstalledEntryMatch(entry, installedServers));
      const name = entry?.name ?? id;

      return {
        id: `integration-${id}`,
        node: (
          <span className={cn(extensionModuleCardPillClassName, "gap-1")}>
            <McpLogoBadge entry={entry} size="xs" />
            {name}
            {installed ? (
              <span className="text-contrast">
                {translate(I18nKey.RECOMMENDED_AUTOMATIONS$CONNECTED)}
              </span>
            ) : !entry ? (
              <span
                className="text-tertiary-alt"
                data-testid={`recommended-automation-integration-${id}`}
              >
                {translate(I18nKey.RECOMMENDED_AUTOMATIONS$UNKNOWN_SETUP)}
              </span>
            ) : !mcpInstallable && !native ? (
              <span
                className="text-tertiary-alt"
                data-testid={`automation-integration-external-${id}`}
              >
                {translate(I18nKey.RECOMMENDED_AUTOMATIONS$EXTERNAL_SETUP)}
              </span>
            ) : null}
          </span>
        ),
      };
    },
  );

  if (missingCount > 0) {
    pills.push({
      id: "missing-connect",
      node: (
        <span className={extensionModuleCardPillClassName}>
          {translate(I18nKey.RECOMMENDED_AUTOMATIONS$MISSING_CONNECT, {
            count: missingCount,
          })}
        </span>
      ),
    });
  }

  return pills;
}

/**
 * A card's badge: the declared glyph when the entry names one, and its
 * integration logos otherwise. Both render into the same slot at the same
 * size, so a card is laid out the same either way.
 */
function AutomationCardIcon({
  automation,
  integrations,
  size,
  testId,
}: {
  automation: RecommendedAutomation;
  integrations: AutomationIntegration[];
  size: "base" | "md";
  testId: string;
}) {
  const Icon = getAutomationIcon(automation);
  if (Icon) {
    return (
      <McpLogoBadge
        entry={null}
        size={size}
        testId={testId}
        fallback={createElement(Icon, {
          className: "h-5 w-5",
          strokeWidth: 2.25,
        })}
      />
    );
  }
  return (
    <McpLogoStackBadge
      entries={integrations.flatMap(({ entry }) => (entry ? [entry] : []))}
      testId={testId}
    />
  );
}

/** The card's trigger-kind mark: the glyph, with the kind's name on hover. */
function TriggerKindBadge({
  kind,
  automationId,
  translate,
}: {
  kind: TemplateTriggerKind;
  automationId: string;
  translate: TFunction;
}) {
  const label = translate(TRIGGER_KIND_LABEL_KEY[kind]);
  const Icon = TRIGGER_KIND_ICON[kind];
  return (
    <StyledTooltip content={label} placement="top">
      <span
        role="img"
        aria-label={label}
        data-testid={`recommended-automation-trigger-${automationId}-${kind}`}
        className="inline-flex size-6 shrink-0 items-center justify-center rounded-md bg-base-secondary text-tertiary-alt"
      >
        <Icon className="size-3.5" aria-hidden />
      </span>
    </StyledTooltip>
  );
}

interface AutomationCardGridProps {
  automations: RecommendedAutomation[];
  installedServers: MCPServerConfig[];
  getNativeIntegration: GetNativeIntegration;
  onSelect: (automation: RecommendedAutomation) => void;
  translate: TFunction;
}

function AutomationCardGrid({
  automations,
  installedServers,
  getNativeIntegration,
  onSelect,
  translate,
}: AutomationCardGridProps) {
  return (
    <div className={cn("mt-3", extensionModuleCardGridClassName)}>
      {automations.map((automation) => {
        const integrations = getIntegrationEntries(automation);
        // "N MCPs to connect" only counts entries the install flow can
        // actually connect (as MCP or natively); an external-setup
        // integration is surfaced on its own pill instead.
        const missingCount = integrations.filter(
          ({ id, entry, mcpInstallable }) => {
            if (!entry || findInstalledEntryMatch(entry, installedServers)) {
              return false;
            }
            const native = getNativeIntegration(id);
            return native ? !native.isConnected : mcpInstallable;
          },
        ).length;

        return (
          <button
            key={automation.id}
            type="button"
            data-testid={`recommended-automation-card-${automation.id}`}
            onClick={() => onSelect(automation)}
            className={cn(
              "flex min-w-0 overflow-hidden p-4 text-left",
              extensionModuleCardSurfaceClassName,
              extensionModuleCardInteractiveClassName,
            )}
          >
            <div className="flex min-w-0 flex-1 items-start gap-3">
              <AutomationCardIcon
                automation={automation}
                integrations={integrations}
                size="md"
                testId={`recommended-automation-icon-${automation.id}`}
              />
              <div className="flex min-w-0 flex-1 flex-col gap-3">
                <header className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <h3 className="truncate text-sm font-semibold text-contrast">
                      {automation.name}
                    </h3>
                    <p className="mt-0.5 truncate text-xs text-tertiary-alt">
                      {automation.category}
                      {!isProvenAutomation(automation) && (
                        <span
                          data-testid={`recommended-automation-beta-${automation.id}`}
                          className="ml-2 rounded border border-border px-1.5 py-px text-[10px] uppercase tracking-wide"
                        >
                          {translate(
                            I18nKey.RECOMMENDED_AUTOMATIONS$BETA_LABEL,
                          )}
                        </span>
                      )}
                    </p>
                  </div>
                  <div className="flex shrink-0 items-center gap-1.5">
                    {getTemplateTriggerKinds(automation).map((kind) => (
                      <TriggerKindBadge
                        key={kind}
                        kind={kind}
                        automationId={automation.id}
                        translate={translate}
                      />
                    ))}
                    <CirclePlusBadge
                      testId={`recommended-automation-plus-${automation.id}`}
                    />
                  </div>
                </header>
                <p className="line-clamp-2 text-xs leading-relaxed text-tertiary-light">
                  {automation.description}
                </p>

                <SkillCardPillRow
                  pills={buildRecommendedAutomationPills(
                    integrations,
                    installedServers,
                    getNativeIntegration,
                    missingCount,
                    translate,
                  )}
                  testId={`recommended-automation-pills-${automation.id}`}
                />
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}

export function RecommendedAutomationsSection({
  backendKind: _backendKind,
  installedServers,
  hiddenAutomationIds,
  unlikelyAutomations,
  suggestedAutomations,
  getNativeIntegration = () => null,
  query = "",
  onClearQuery,
  filters,
  onFiltersChange,
  onSelect,
  scrollableGrid = false,
}: RecommendedAutomationsSectionProps) {
  const { t } = useTranslation("openhands");

  // Only the deployment's hide list, the filters and the query narrow the
  // grid. An automation that declares no integration needs nothing connected,
  // and one naming an integration this host cannot resolve is shown with that
  // gap on its pill, so neither is a reason to hide a card the catalog ships.
  const offeredAutomations = RECOMMENDED_AUTOMATIONS.filter(
    (automation) => !hiddenAutomationIds?.has(automation.id),
  );
  const visibleAutomations = offeredAutomations.filter(
    (automation) =>
      (!filters || automationMatchesFilters(automation, filters)) &&
      automationMatchesQuery(
        automation,
        getIntegrationEntries(automation),
        query,
      ),
  );

  const filterChips =
    filters && onFiltersChange ? (
      <TemplateFilterChips
        integrationIds={collectIntegrationIds(offeredAutomations)}
        filters={filters}
        onChange={onFiltersChange}
      />
    ) : null;

  if (visibleAutomations.length === 0) {
    // A search or filter that matches nothing says so, with a way back. A
    // surface without either control keeps rendering nothing.
    const canClear = !!onClearQuery || !!onFiltersChange;
    const isNarrowed =
      query.trim().length > 0 ||
      (!!filters && hasActiveTemplateFilter(filters));
    if (!canClear || !isNarrowed) return null;
    return (
      <section
        data-testid="recommended-automations-section"
        className={cn("flex flex-col gap-4", filterChips ? "mt-0" : "mt-3")}
      >
        {filterChips}
        <AutomationsFilteredEmptyState
          message={t(I18nKey.RECOMMENDED_AUTOMATIONS$NO_MATCHING_TEMPLATES)}
          onClear={() => {
            onClearQuery?.();
            onFiltersChange?.(EMPTY_TEMPLATE_FILTERS);
          }}
        />
      </section>
    );
  }

  const { suggested, more } = splitBySuggestion(
    visibleAutomations,
    isProvenAutomation,
    { suggested: suggestedAutomations, unlikely: unlikelyAutomations },
  );

  return (
    <section
      data-testid="recommended-automations-section"
      className={cn(scrollableGrid && "flex min-h-0 flex-1 flex-col")}
    >
      <div
        data-testid={
          scrollableGrid ? "recommended-automations-scroll-area" : undefined
        }
        className={cn(
          filterChips ? "mt-0" : "mt-3",
          extensionModuleCardGridContainerClassName,
          scrollableGrid &&
            "min-h-0 flex-1 overflow-y-auto custom-scrollbar-always",
        )}
      >
        {filterChips && <div className="mb-4">{filterChips}</div>}

        {suggested.length > 0 && (
          <>
            <div className="flex items-center">
              <h2 className="text-base font-semibold text-foreground">
                {t(I18nKey.RECOMMENDED_AUTOMATIONS$SUGGESTED_TITLE)}
              </h2>
              <StatusBadge count={suggested.length} />
            </div>
            <p className="mt-1 text-sm text-muted">
              {t(I18nKey.RECOMMENDED_AUTOMATIONS$SECTION_DESCRIPTION)}
            </p>

            <AutomationCardGrid
              automations={suggested}
              installedServers={installedServers}
              getNativeIntegration={getNativeIntegration}
              onSelect={onSelect}
              translate={t}
            />
          </>
        )}

        {more.length > 0 && (
          <section
            data-testid="recommended-automations-more-section"
            className={cn(suggested.length > 0 && "mt-8")}
          >
            <div
              data-testid="recommended-automations-more-heading"
              className="flex items-center"
            >
              <h2 className="text-base font-semibold text-foreground">
                {t(I18nKey.RECOMMENDED_AUTOMATIONS$MORE_TITLE)}
              </h2>
              <StatusBadge count={more.length} />
            </div>

            <AutomationCardGrid
              automations={more}
              installedServers={installedServers}
              getNativeIntegration={getNativeIntegration}
              onSelect={onSelect}
              translate={t}
            />
          </section>
        )}
      </div>
    </section>
  );
}
