import type {
  DependencyTrackIntegrationConfig,
  DependencyTrackIntegrationsSettings,
  IntegrationsSettings,
} from "#/types/integrations";
import { dependencyTrackApiKeySecretName } from "#/utils/dependency-track-integration-secrets";
import { findWorkspaceIdForPath } from "#/utils/appwrite-workspace-config";

export const EMPTY_DEPENDENCY_TRACK_CONFIG: DependencyTrackIntegrationConfig = {
  enabled: false,
  baseUrl: "",
  projectUuid: "",
};

export function normalizeDependencyTrackIntegrations(
  raw: unknown,
): DependencyTrackIntegrationsSettings {
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
      byWorkspace: record.byWorkspace as Record<
        string,
        DependencyTrackIntegrationConfig
      >,
    };
  }
  return { byWorkspace: {} };
}

export function getDependencyTrackConfigForWorkspace(
  integrations: IntegrationsSettings | undefined,
  workspaceId: string | null | undefined,
): DependencyTrackIntegrationConfig {
  if (!workspaceId) {
    return { ...EMPTY_DEPENDENCY_TRACK_CONFIG };
  }
  const normalized = normalizeDependencyTrackIntegrations(
    integrations?.dependencyTrack,
  );
  const stored = normalized.byWorkspace[workspaceId];
  if (!stored) {
    return { ...EMPTY_DEPENDENCY_TRACK_CONFIG };
  }
  return {
    enabled: Boolean(stored.enabled),
    baseUrl: stored.baseUrl?.trim() || "",
    projectUuid: stored.projectUuid?.trim() || "",
    apiKeySecretName: stored.apiKeySecretName,
  };
}

export function resolveDependencyTrackSecretName(
  config: DependencyTrackIntegrationConfig,
  workspaceId: string,
): string {
  return (
    config.apiKeySecretName?.trim() ||
    dependencyTrackApiKeySecretName(workspaceId)
  );
}

export { findWorkspaceIdForPath };

export function buildDependencyTrackIntegrationsPatch(
  current: IntegrationsSettings | undefined,
  workspaceId: string,
  config: DependencyTrackIntegrationConfig,
): IntegrationsSettings {
  const normalized = normalizeDependencyTrackIntegrations(
    current?.dependencyTrack,
  );
  return {
    dependencyTrack: {
      byWorkspace: {
        ...normalized.byWorkspace,
        [workspaceId]: {
          enabled: config.enabled,
          baseUrl: config.baseUrl.trim(),
          projectUuid: config.projectUuid.trim(),
          apiKeySecretName:
            config.apiKeySecretName?.trim() ||
            dependencyTrackApiKeySecretName(workspaceId),
        },
      },
    },
  };
}
