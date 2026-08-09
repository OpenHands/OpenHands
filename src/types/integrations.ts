/**
 * Frontend-owned integration configs stored under
 * `misc_settings.integrations` on the agent-server.
 *
 * Each provider is scoped per local workspace: each workspace id maps to its
 * own endpoint / project / secret name.
 */

export type AppwriteIntegrationConfig = {
  enabled: boolean;
  /** AppWrite API base including `/v1`, e.g. `https://cloud.appwrite.io/v1`. */
  endpoint: string;
  projectId: string;
  /** Overrides the default per-workspace secret name when set. */
  apiKeySecretName?: string;
};

export type AppwriteIntegrationsSettings = {
  /** Map of local workspace id → AppWrite project config. */
  byWorkspace: Record<string, AppwriteIntegrationConfig>;
};

export type PlaneIntegrationConfig = {
  enabled: boolean;
  /** Self-hosted or cloud Plane base URL, e.g. `https://plane.example.com`. */
  baseUrl: string;
  /** Plane workspace slug used in `/api/v1/workspaces/{slug}/…`. */
  workspaceSlug: string;
  projectId: string;
  /** Optional Plane module id when the project is associated with a module. */
  moduleId?: string;
  /** Overrides the default per-workspace secret name when set. */
  apiKeySecretName?: string;
};

export type PlaneIntegrationsSettings = {
  /** Map of local workspace id → Plane project config. */
  byWorkspace: Record<string, PlaneIntegrationConfig>;
};

export type DependencyTrackIntegrationConfig = {
  enabled: boolean;
  /** Dependency-Track base URL, e.g. `https://dtrack.example.com`. */
  baseUrl: string;
  /** Target project UUID in Dependency-Track. */
  projectUuid: string;
  apiKeySecretName?: string;
};

export type DependencyTrackIntegrationsSettings = {
  byWorkspace: Record<string, DependencyTrackIntegrationConfig>;
};

export type IntegrationsSettings = {
  appwrite?: AppwriteIntegrationsSettings;
  plane?: PlaneIntegrationsSettings;
  dependencyTrack?: DependencyTrackIntegrationsSettings;
};
