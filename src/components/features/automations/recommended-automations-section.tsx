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
import { MCPServerConfig } from "#/types/mcp-server";
import {
  findInstalledEntryMatch,
  getMarketplaceEntryById,
  isMcpInstallableEntry,
} from "#/utils/mcp-marketplace-utils";
import { getFeaturedAutomationIds } from "#/manifests/automation-interface";
import {
  getAutomationLaunchPrompt,
  getIntegrationIds,
} from "#/utils/automation-catalog";
import { cn } from "#/utils/utils";
import {
  extensionModuleCardInteractiveClassName,
  extensionModuleCardGridClassName,
  extensionModuleCardGridContainerClassName,
  extensionModuleCardPillClassName,
} from "#/utils/extension-module-card-classes";
import { StatusBadge } from "./status-badge";

interface RecommendedAutomationsSectionProps {
  backendKind: "local" | "cloud";
  installedServers: MCPServerConfig[];
  query?: string;
  onSelect: (automation: RecommendedAutomation) => void;
  /** When true, title, description, and cards share one scroll area. */
  scrollableGrid?: boolean;
}

export function getAutomationsByPopularity(
  catalog: RecommendedAutomation[],
): RecommendedAutomation[] {
  return catalog
    .map((automation, index) => ({ automation, index }))
    .sort((a, b) => {
      const byPopularity =
        (b.automation.popularityRank ?? 0) - (a.automation.popularityRank ?? 0);
      return byPopularity || a.index - b.index;
    })
    .map(({ automation }) => automation);
}

const RECOMMENDED_AUTOMATIONS = getAutomationsByPopularity(AUTOMATION_CATALOG);

// Proven automations are featured above the Beta group. The set is owned by
// the interface manifest (falling back to the host default), not derived from
// popularityRank: slack-standup-digest@94 outranks slack-channel-monitor@92
// yet is Beta.
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

/**
 * Keep any automation with declared integrations visible, including when a
 * catalog entry is missing. This makes catalog drift actionable instead of
 * silently hiding the automation.
 */
function isAutomationAvailable(automation: RecommendedAutomation) {
  return getIntegrationIds(automation).length > 0;
}

function buildRecommendedAutomationPills(
  integrations: AutomationIntegration[],
  installedServers: MCPServerConfig[],
  missingCount: number,
  translate: TFunction,
): SkillCardPill[] {
  const pills: SkillCardPill[] = integrations.map(
    ({ id, entry, mcpInstallable }) => {
      const installed =
        !!entry && findInstalledEntryMatch(entry, installedServers);
      const name = entry?.name ?? id;

      return {
        id: `integration-${id}`,
        node: (
          <span className={cn(extensionModuleCardPillClassName, "gap-1")}>
            <McpLogoBadge entry={entry} size="xs" />
            {name}
            {installed ? (
              <span className="text-white">
                {translate(I18nKey.RECOMMENDED_AUTOMATIONS$CONNECTED)}
              </span>
            ) : !entry ? (
              <span
                className="text-tertiary-alt"
                data-testid={`recommended-automation-integration-${id}`}
              >
                {translate(I18nKey.RECOMMENDED_AUTOMATIONS$UNKNOWN_SETUP)}
              </span>
            ) : !mcpInstallable ? (
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

interface AutomationCardGridProps {
  automations: RecommendedAutomation[];
  installedServers: MCPServerConfig[];
  onSelect: (automation: RecommendedAutomation) => void;
  translate: TFunction;
}

function AutomationCardGrid({
  automations,
  installedServers,
  onSelect,
  translate,
}: AutomationCardGridProps) {
  return (
    <div className={cn("mt-3", extensionModuleCardGridClassName)}>
      {automations.map((automation) => {
        const integrations = getIntegrationEntries(automation);
        // "N MCPs to connect" only counts entries the install flow can
        // actually connect; an external-setup integration is surfaced on its
        // own pill instead.
        const missingCount = integrations.filter(
          ({ entry, mcpInstallable }) =>
            !!entry &&
            mcpInstallable &&
            !findInstalledEntryMatch(entry, installedServers),
        ).length;

        return (
          <button
            key={automation.id}
            type="button"
            data-testid={`recommended-automation-card-${automation.id}`}
            onClick={() => onSelect(automation)}
            className={cn(
              "flex min-w-0 overflow-hidden p-4 text-left rounded-xl bg-surface-raised",
              extensionModuleCardInteractiveClassName,
            )}
          >
            <div className="flex min-w-0 flex-1 items-start gap-3">
              <McpLogoStackBadge
                entries={integrations.flatMap(({ entry }) =>
                  entry ? [entry] : [],
                )}
                testId={`recommended-automation-icon-${automation.id}`}
              />
              <div className="flex min-w-0 flex-1 flex-col gap-3">
                <header className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <h3 className="truncate text-sm font-semibold text-white">
                      {automation.name}
                    </h3>
                    <p className="mt-0.5 truncate text-xs text-tertiary-alt">
                      {automation.category}
                    </p>
                  </div>
                  <CirclePlusBadge
                    testId={`recommended-automation-plus-${automation.id}`}
                  />
                </header>
                <p className="line-clamp-2 text-xs leading-relaxed text-tertiary-light">
                  {automation.description}
                </p>

                <SkillCardPillRow
                  pills={buildRecommendedAutomationPills(
                    integrations,
                    installedServers,
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
  query = "",
  onSelect,
  scrollableGrid = false,
}: RecommendedAutomationsSectionProps) {
  const { t } = useTranslation("openhands");

  const visibleAutomations = RECOMMENDED_AUTOMATIONS.filter((automation) => {
    const integrationEntries = getIntegrationEntries(automation);
    return (
      isAutomationAvailable(automation) &&
      automationMatchesQuery(automation, integrationEntries, query)
    );
  });

  if (visibleAutomations.length === 0) return null;

  const provenAutomations = visibleAutomations.filter(isProvenAutomation);
  const betaAutomations = visibleAutomations.filter(
    (automation) => !isProvenAutomation(automation),
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
          "mt-3",
          extensionModuleCardGridContainerClassName,
          scrollableGrid &&
            "min-h-0 flex-1 overflow-y-auto custom-scrollbar-always",
        )}
      >
        {provenAutomations.length > 0 && (
          <>
            <div className="flex items-center">
              <h2 className="text-base font-semibold text-foreground">
                {t(I18nKey.RECOMMENDED_AUTOMATIONS$SECTION_TITLE)}
              </h2>
              <StatusBadge count={provenAutomations.length} />
            </div>
            <p className="mt-1 text-sm text-muted">
              {t(I18nKey.RECOMMENDED_AUTOMATIONS$SECTION_DESCRIPTION)}
            </p>

            <AutomationCardGrid
              automations={provenAutomations}
              installedServers={installedServers}
              onSelect={onSelect}
              translate={t}
            />
          </>
        )}

        {betaAutomations.length > 0 && (
          <section
            data-testid="recommended-automations-beta-section"
            className={cn(provenAutomations.length > 0 && "mt-8")}
          >
            <div
              data-testid="recommended-automations-beta-heading"
              className="flex items-center"
            >
              <h2 className="text-base font-semibold text-foreground">
                {t(I18nKey.RECOMMENDED_AUTOMATIONS$BETA_LABEL)}
              </h2>
              <StatusBadge count={betaAutomations.length} />
            </div>

            <AutomationCardGrid
              automations={betaAutomations}
              installedServers={installedServers}
              onSelect={onSelect}
              translate={t}
            />
          </section>
        )}
      </div>
    </section>
  );
}
