/**
 * AppWrite integration secrets are scoped per workspace so each workspace
 * can point at a different AppWrite project with its own API key.
 */

export const DEFAULT_APPWRITE_ENDPOINT = "https://cloud.appwrite.io/v1";

/** Header the frontend sends so the Canvas proxy can pick the right config. */
export const APPWRITE_WORKSPACE_ID_HEADER = "X-OpenHands-Workspace-Id";

/**
 * Sanitize a workspace id into a secrets-store-safe name fragment
 * (letters, digits, underscores; 1–48 chars).
 */
export function sanitizeWorkspaceIdForSecret(workspaceId: string): string {
  const cleaned = workspaceId
    .trim()
    .replace(/[^A-Za-z0-9_]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48);
  return cleaned || "workspace";
}

export function appwriteApiKeySecretName(workspaceId: string): string {
  return `INTEGRATION_APPWRITE_API_KEY_${sanitizeWorkspaceIdForSecret(workspaceId)}`;
}
