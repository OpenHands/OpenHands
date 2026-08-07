/**
 * Frontend-owned integration configs stored under
 * `misc_settings.integrations` on the agent-server.
 *
 * AppWrite is scoped per workspace: each workspace id maps to its own
 * endpoint / project / secret name.
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

export type IntegrationsSettings = {
  appwrite?: AppwriteIntegrationsSettings;
};
