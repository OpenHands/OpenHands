import type {
  IntegrationsSettings,
  PlaneIntegrationConfig,
  PlaneIntegrationsSettings,
} from "#/types/integrations";
import {
  DEFAULT_PLANE_BASE_URL,
  planeApiKeySecretName,
} from "#/utils/plane-integration-secrets";

export const EMPTY_PLANE_CONFIG: PlaneIntegrationConfig = {
  enabled: false,
  baseUrl: DEFAULT_PLANE_BASE_URL,
  workspaceSlug: "",
  projectId: "",
  moduleId: "",
};

/**
 * Normalize stored Plane settings to the per-workspace shape.
 */
export function normalizePlaneIntegrations(
  raw: unknown,
): PlaneIntegrationsSettings {
  if (!raw || typeof raw !== "object") {
    return { byWorkspace: {} };
  }
  const record = raw as Record<string, unknown>;
  if (
    record.byWorkspace &&
    typeof record.byWorkspace === "object" &&
    !Array.isArray(record.byWorkspace)
  ) {
    return {
      byWorkspace: record.byWorkspace as Record<string, PlaneIntegrationConfig>,
    };
  }
  return { byWorkspace: {} };
}

export function getPlaneConfigForWorkspace(
  integrations: IntegrationsSettings | undefined,
  workspaceId: string | null | undefined,
): PlaneIntegrationConfig {
  if (!workspaceId) {
    return { ...EMPTY_PLANE_CONFIG };
  }
  const normalized = normalizePlaneIntegrations(integrations?.plane);
  const stored = normalized.byWorkspace[workspaceId];
  if (!stored) {
    return { ...EMPTY_PLANE_CONFIG };
  }
  return {
    enabled: Boolean(stored.enabled),
    baseUrl: stored.baseUrl?.trim() || DEFAULT_PLANE_BASE_URL,
    workspaceSlug: stored.workspaceSlug?.trim() || "",
    projectId: stored.projectId?.trim() || "",
    moduleId: stored.moduleId?.trim() || "",
    apiKeySecretName: stored.apiKeySecretName,
  };
}

export function resolvePlaneSecretName(
  config: PlaneIntegrationConfig,
  workspaceId: string,
): string {
  return config.apiKeySecretName?.trim() || planeApiKeySecretName(workspaceId);
}

export function buildPlaneIntegrationsPatch(
  current: IntegrationsSettings | undefined,
  workspaceId: string,
  config: PlaneIntegrationConfig,
): IntegrationsSettings {
  const normalized = normalizePlaneIntegrations(current?.plane);
  const moduleId = config.moduleId?.trim() || undefined;
  return {
    plane: {
      byWorkspace: {
        ...normalized.byWorkspace,
        [workspaceId]: {
          enabled: config.enabled,
          baseUrl: config.baseUrl.trim(),
          workspaceSlug: config.workspaceSlug.trim(),
          projectId: config.projectId.trim(),
          ...(moduleId ? { moduleId } : {}),
          apiKeySecretName:
            config.apiKeySecretName?.trim() || planeApiKeySecretName(workspaceId),
        },
      },
    },
  };
}
