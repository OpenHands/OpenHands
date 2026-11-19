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
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useSearchSecrets } from "./use-get-secrets";
import { useSettings } from "./use-settings";
import { useNativeGitIntegrations } from "./use-native-git-integrations";

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
  const { backend } = useActiveBackend();
  const { data: settings, isLoading } = useSettings();
  const { data: secrets, isLoading: isLoadingSecrets } = useSearchSecrets({
    enabled: Object.hasOwn(integrations, "gitlab"),
  });
  const { getNativeIntegration, isLoading: isNativeIntegrationsLoading } =
    useNativeGitIntegrations();

  const installedServers = useMemo(
    () =>
      flattenMcpConfig(
        settings?.mcp_config ??
          parseMcpConfig(settings?.agent_settings?.mcp_config),
      ).filter((server) => server.enabled !== false),
    [settings?.agent_settings?.mcp_config, settings?.mcp_config],
  );

  const missingIntegrations = useMemo<MissingSetupIntegration[]>(() => {
    const catalog = getMcpMarketplaceCatalog(MCP_MARKETPLACE);
    const hasGitLabToken =
      secrets.some(
        ({ name }) =>
          name === "GITLAB_TOKEN" ||
          (backend.kind === "cloud" && name === "gitlab_token"),
      ) ||
      (backend.kind === "cloud" &&
        Object.hasOwn(settings?.provider_tokens_set ?? {}, "gitlab"));
    return Object.entries(integrations)
      .map(([id, requirement]) => ({
        id,
        requirement,
        entry: getMarketplaceEntryById(id, catalog) ?? null,
      }))
      .filter(({ id, entry: catalogEntry }) => {
        if (getNativeIntegration(id)?.isConnected) return false;
        if (id === "gitlab") {
          return (
            !hasGitLabToken &&
            (backend.kind !== "cloud" ||
              !catalogEntry ||
              !findInstalledEntryMatch(catalogEntry, installedServers))
          );
        }
        return (
          !catalogEntry ||
          !findInstalledEntryMatch(catalogEntry, installedServers)
        );
      });
  }, [
    backend.kind,
    getNativeIntegration,
    integrations,
    installedServers,
    secrets,
    settings?.provider_tokens_set,
  ]);

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
    isLoading: isLoading || isNativeIntegrationsLoading || isLoadingSecrets,
  };
}
