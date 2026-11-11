import { useCloudIntegrationsConfig } from "#/hooks/query/use-cloud-integrations-config";

/** Preserve local/Cloud defaults when the instance has not disabled selection. */
export function useAgentProfileAvailability(): boolean {
  const { data: config } = useCloudIntegrationsConfig();
  return !(
    config?.deploymentMode === "self_hosted" && !config.isAgentProfilesEnabled
  );
}
