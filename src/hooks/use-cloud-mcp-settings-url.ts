import { useActiveBackend } from "#/contexts/active-backend-context";
import { useCloudIntegrationsConfig } from "#/hooks/query/use-cloud-integrations-config";

/**
 * The cloud instance's Settings > MCP page when MCP servers are managed there
 * instead of on the canvas MCP page: an OpenHands Enterprise (self-hosted)
 * instance with organization MCP servers on, whose settings page is the only
 * one that lists them (OHE-3869). `url` is null otherwise, including while the
 * instance config is loading or could not be read. `org` makes the settings
 * page open on the org that is active here.
 */
export function useCloudMcpSettingsUrl(): {
  url: string | null;
  isLoading: boolean;
} {
  const { backend, orgId } = useActiveBackend();
  const { data: config, isLoading } = useCloudIntegrationsConfig();

  if (
    backend.kind !== "cloud" ||
    config?.deploymentMode !== "self_hosted" ||
    !config.isOrgMcpServersEnabled
  ) {
    return { url: null, isLoading };
  }
  const orgQuery = orgId ? `?org=${encodeURIComponent(orgId)}` : "";
  return {
    url: `${backend.host.replace(/\/+$/, "")}/settings/mcp${orgQuery}`,
    isLoading,
  };
}
