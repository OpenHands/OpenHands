import { getLockedCloudHost } from "#/api/agent-server-config";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useCloudIntegrationsConfig } from "#/hooks/query/use-cloud-integrations-config";

/**
 * Whether Enterprise callouts apply to the active backend. Open source
 * backends and OpenHands Cloud (which has no paid tier) see them; a
 * self-hosted Enterprise instance, or a Canvas locked to a Cloud host, never
 * does. Stays false while a Cloud backend's deployment mode is still loading,
 * so nothing flashes in and out.
 */
export function useEnterpriseCalloutsVisible(): boolean {
  const { backend } = useActiveBackend();
  const { data: config, isLoading } = useCloudIntegrationsConfig();

  if (getLockedCloudHost() !== null) return false;
  if (backend.kind === "local") return true;
  if (isLoading) return false;
  return config?.deploymentMode !== "self_hosted";
}
