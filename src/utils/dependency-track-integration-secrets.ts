/**
 * Dependency-Track integration secrets are scoped per workspace.
 */

/** Reuse the shared workspace header for integration proxies. */
export { APPWRITE_WORKSPACE_ID_HEADER as INTEGRATION_WORKSPACE_ID_HEADER } from "#/utils/appwrite-integration-secrets";

export function sanitizeWorkspaceIdForSecret(workspaceId: string): string {
  const cleaned = workspaceId
    .trim()
    .replace(/[^A-Za-z0-9_]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48);
  return cleaned || "workspace";
}

export function dependencyTrackApiKeySecretName(workspaceId: string): string {
  return `INTEGRATION_DEPENDENCY_TRACK_API_KEY_${sanitizeWorkspaceIdForSecret(workspaceId)}`;
}
