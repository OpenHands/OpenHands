import { useMemo } from "react";
import { useSettings } from "#/hooks/query/use-settings";
import { useSearchSecrets } from "#/hooks/query/use-get-secrets";
import type { PlaneIntegrationConfig } from "#/types/integrations";
import {
  EMPTY_PLANE_CONFIG,
  getPlaneConfigForWorkspace,
  resolvePlaneSecretName,
} from "#/utils/plane-workspace-config";

export type PlaneIntegrationState = {
  workspaceId: string | null;
  config: PlaneIntegrationConfig;
  apiKeyIsSet: boolean;
  /** True when a workspace is selected and its Plane config is complete. */
  isReady: boolean;
  isLoading: boolean;
  secretName: string | null;
};

/**
 * Reads Plane integration config for a specific local workspace id.
 */
export function usePlaneIntegration(
  workspaceId: string | null | undefined,
): PlaneIntegrationState {
  const { data: settings, isLoading: settingsLoading } = useSettings();
  const { data: secrets, isLoading: secretsLoading } = useSearchSecrets();

  return useMemo(() => {
    const id = workspaceId?.trim() || null;
    const config = getPlaneConfigForWorkspace(settings?.integrations, id);
    const secretName = id ? resolvePlaneSecretName(config, id) : null;
    const apiKeyIsSet = Boolean(
      secretName && (secrets ?? []).some((s) => s.name === secretName),
    );
    const isReady =
      Boolean(id) &&
      config.enabled &&
      Boolean(config.baseUrl) &&
      Boolean(config.workspaceSlug) &&
      Boolean(config.projectId) &&
      apiKeyIsSet;

    return {
      workspaceId: id,
      config: id ? config : { ...EMPTY_PLANE_CONFIG },
      apiKeyIsSet,
      isReady,
      isLoading: settingsLoading || secretsLoading,
      secretName,
    };
  }, [workspaceId, settings, secrets, settingsLoading, secretsLoading]);
}
