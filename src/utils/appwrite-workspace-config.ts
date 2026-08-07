import type {
  AppwriteIntegrationConfig,
  AppwriteIntegrationsSettings,
  IntegrationsSettings,
} from "#/types/integrations";
import type { LocalWorkspace } from "#/types/workspace";
import {
  DEFAULT_APPWRITE_ENDPOINT,
  appwriteApiKeySecretName,
} from "#/utils/appwrite-integration-secrets";

export const EMPTY_APPWRITE_CONFIG: AppwriteIntegrationConfig = {
  enabled: false,
  endpoint: DEFAULT_APPWRITE_ENDPOINT,
  projectId: "",
};

/**
 * Normalize stored AppWrite settings. Accepts the per-workspace shape and
 * silently ignores the legacy global single-project shape.
 */
export function normalizeAppwriteIntegrations(
  raw: unknown,
): AppwriteIntegrationsSettings {
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
        AppwriteIntegrationConfig
      >,
    };
  }
  return { byWorkspace: {} };
}

export function getAppwriteConfigForWorkspace(
  integrations: IntegrationsSettings | undefined,
  workspaceId: string | null | undefined,
): AppwriteIntegrationConfig {
  if (!workspaceId) {
    return { ...EMPTY_APPWRITE_CONFIG };
  }
  const normalized = normalizeAppwriteIntegrations(integrations?.appwrite);
  const stored = normalized.byWorkspace[workspaceId];
  if (!stored) {
    return { ...EMPTY_APPWRITE_CONFIG };
  }
  return {
    enabled: Boolean(stored.enabled),
    endpoint: stored.endpoint?.trim() || DEFAULT_APPWRITE_ENDPOINT,
    projectId: stored.projectId?.trim() || "",
    apiKeySecretName: stored.apiKeySecretName,
  };
}

export function resolveAppwriteSecretName(
  config: AppwriteIntegrationConfig,
  workspaceId: string,
): string {
  return (
    config.apiKeySecretName?.trim() || appwriteApiKeySecretName(workspaceId)
  );
}

/**
 * Match a conversation's attached workspace path / working_dir to a
 * registered local workspace id.
 */
export function findWorkspaceIdForPath(
  workspaces: LocalWorkspace[],
  path: string | null | undefined,
): string | null {
  const candidate = path?.trim();
  if (!candidate || workspaces.length === 0) {
    return null;
  }

  const normalize = (value: string) =>
    value.replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
  const normalizedCandidate = normalize(candidate);

  const exact = workspaces.find(
    (workspace) => normalize(workspace.path) === normalizedCandidate,
  );
  if (exact) {
    return exact.id;
  }

  // Worktrees / nested dirs: longest registered path that is a prefix.
  const byPrefix = [...workspaces]
    .map((workspace) => ({
      workspace,
      normalized: normalize(workspace.path),
    }))
    .filter(
      ({ normalized }) =>
        normalizedCandidate === normalized ||
        normalizedCandidate.startsWith(`${normalized}/`),
    )
    .sort((a, b) => b.normalized.length - a.normalized.length);

  return byPrefix[0]?.workspace.id ?? null;
}

export function buildAppwriteIntegrationsPatch(
  current: IntegrationsSettings | undefined,
  workspaceId: string,
  config: AppwriteIntegrationConfig,
): IntegrationsSettings {
  const normalized = normalizeAppwriteIntegrations(current?.appwrite);
  return {
    appwrite: {
      byWorkspace: {
        ...normalized.byWorkspace,
        [workspaceId]: {
          enabled: config.enabled,
          endpoint: config.endpoint.trim(),
          projectId: config.projectId.trim(),
          apiKeySecretName:
            config.apiKeySecretName?.trim() ||
            appwriteApiKeySecretName(workspaceId),
        },
      },
    },
  };
}
