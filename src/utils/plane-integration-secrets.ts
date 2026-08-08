/**
 * Plane integration secrets are scoped per workspace so each local workspace
 * can point at a different Plane project with its own API key.
 */

import { sanitizeWorkspaceIdForSecret } from "#/utils/appwrite-integration-secrets";

/** Empty by default — self-hosted deployments supply their own URL. */
export const DEFAULT_PLANE_BASE_URL = "";

/** Header the frontend sends so the Canvas proxy can pick the right config. */
export const PLANE_WORKSPACE_ID_HEADER = "X-OpenHands-Workspace-Id";

export function planeApiKeySecretName(workspaceId: string): string {
  return `INTEGRATION_PLANE_API_KEY_${sanitizeWorkspaceIdForSecret(workspaceId)}`;
}
