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
  ExtensionManifest,
  ManifestIntegrationRequirement,
  ManifestSecretRequirement,
} from "#/manifests/types";
import { useSettings } from "./use-settings";
import { useSearchSecrets } from "./use-get-secrets";

export interface MissingManifestIntegration {
  requirement: ManifestIntegrationRequirement;
  /** The catalog entry, when the id resolves to one. */
  entry: MarketplaceEntry | null;
}

export interface ManifestPrerequisitesResult {
  /** Unconnected integrations the manifest declares as blocking. */
  blockingIntegrations: MissingManifestIntegration[];
  /** Unconnected integrations the manifest is willing to proceed without. */
  warningIntegrations: MissingManifestIntegration[];
  /** Required credentials the deployment does not yet hold. */
  missingSecrets: ManifestSecretRequirement[];
  isBlocked: boolean;
  isLoading: boolean;
}

/**
 * Stages 3 and 4 — which accounts are connected and which credentials exist.
 *
 * Credentials are observed by *name only*. The host learns whether a credential
 * is present; it never reads, collects, or forwards its value, so a manifest
 * naming a secret can never become a route by which the host handles one.
 */
export function useManifestPrerequisites(
  manifest: ExtensionManifest,
): ManifestPrerequisitesResult {
  const requires = manifest.requires;
  const { data: settings, isLoading: isLoadingSettings } = useSettings();
  const { data: secrets, isLoading: isLoadingSecrets } = useSearchSecrets({
    enabled: (requires?.secrets.length ?? 0) > 0,
  });

  const installedServers = useMemo(
    () =>
      flattenMcpConfig(parseMcpConfig(settings?.agent_settings?.mcp_config)),
    [settings?.agent_settings?.mcp_config],
  );

  const missingIntegrations = useMemo<MissingManifestIntegration[]>(() => {
    const catalog = getMcpMarketplaceCatalog(MCP_MARKETPLACE);
    return (requires?.integrations ?? [])
      .map((requirement) => ({
        requirement,
        entry: getMarketplaceEntryById(requirement.id, catalog) ?? null,
      }))
      .filter(
        ({ entry }) =>
          !entry || !findInstalledEntryMatch(entry, installedServers),
      );
  }, [requires?.integrations, installedServers]);

  const missingSecrets = useMemo(() => {
    const present = new Set((secrets ?? []).map((secret) => secret.name));
    return (requires?.secrets ?? []).filter(
      (secret) => secret.required && !present.has(secret.key),
    );
  }, [requires?.secrets, secrets]);

  const blockingIntegrations = missingIntegrations.filter(
    ({ requirement }) => requirement.enforcement === "block",
  );
  const warningIntegrations = missingIntegrations.filter(
    ({ requirement }) => requirement.enforcement === "warn",
  );

  return {
    blockingIntegrations,
    warningIntegrations,
    missingSecrets,
    isBlocked: blockingIntegrations.length > 0 || missingSecrets.length > 0,
    isLoading: isLoadingSettings || isLoadingSecrets,
  };
}
