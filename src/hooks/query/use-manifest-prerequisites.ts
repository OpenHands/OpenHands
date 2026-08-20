import { useMemo } from "react";
import {
  INTEGRATION_CATALOG as MCP_MARKETPLACE,
  type IntegrationCatalogEntry as MarketplaceEntry,
} from "@openhands/extensions/integrations";
import {
  findInstalledEntryMatch,
  getMarketplaceEntryById,
  getMcpMarketplaceCatalog,
} from "#/utils/mcp-marketplace-utils";
import { flattenMcpConfig } from "#/utils/mcp-installed-servers";
import { parseMcpConfig } from "#/utils/mcp-config";
import type {
  SetupEntry,
  SetupIntegrationRequirement,
} from "#/manifests/types";
import { useSettings } from "./use-settings";

export interface MissingSetupIntegration {
  id: string;
  requirement: SetupIntegrationRequirement;
  /** The catalog entry, when the id resolves to one. */
  entry: MarketplaceEntry | null;
}

export interface SetupPrerequisitesResult {
  /** Unconnected integrations the manifest declares as required. */
  blockingIntegrations: MissingSetupIntegration[];
  /** Unconnected integrations the manifest is willing to proceed without. */
  warningIntegrations: MissingSetupIntegration[];
  isBlocked: boolean;
  isLoading: boolean;
}

/**
 * Stage 3 — which of the accounts this manifest needs are already connected.
 *
 * A requirement is blocking unless it opts out with `required: false`, which is
 * the manifest's way of saying the integration can be connected later, during
 * setup itself.
 */
export function useSetupPrerequisites(
  entry: SetupEntry,
): SetupPrerequisitesResult {
  const integrations = entry.requires.integrations;
  const { data: settings, isLoading } = useSettings();

  const installedServers = useMemo(
    () =>
      flattenMcpConfig(parseMcpConfig(settings?.agent_settings?.mcp_config)),
    [settings?.agent_settings?.mcp_config],
  );

  const missingIntegrations = useMemo<MissingSetupIntegration[]>(() => {
    const catalog = getMcpMarketplaceCatalog(MCP_MARKETPLACE);
    return Object.entries(integrations)
      .map(([id, requirement]) => ({
        id,
        requirement,
        entry: getMarketplaceEntryById(id, catalog) ?? null,
      }))
      .filter(
        ({ entry: catalogEntry }) =>
          !catalogEntry ||
          !findInstalledEntryMatch(catalogEntry, installedServers),
      );
  }, [integrations, installedServers]);

  const blockingIntegrations = missingIntegrations.filter(
    ({ requirement }) => requirement.required !== false,
  );
  const warningIntegrations = missingIntegrations.filter(
    ({ requirement }) => requirement.required === false,
  );

  return {
    blockingIntegrations,
    warningIntegrations,
    isBlocked: blockingIntegrations.length > 0,
    isLoading,
  };
}
