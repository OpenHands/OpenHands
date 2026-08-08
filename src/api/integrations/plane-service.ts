import axios from "axios";
import { NoBackendAvailableError } from "#/api/agent-server-client-options";
import { getEffectiveLocalBackend } from "#/api/backend-registry/active-store";
import { PLANE_WORKSPACE_ID_HEADER } from "#/utils/plane-integration-secrets";

/**
 * Canvas-owned Plane proxy path. Requests go to ingress/static-server,
 * which resolves the API key server-side from the Secrets store for the
 * workspace identified by {@link PLANE_WORKSPACE_ID_HEADER}.
 */
export const PLANE_PROXY_BASE = "/api/integrations/plane";

async function planeRequest<T>(
  workspaceId: string,
  method: string,
  path: string,
): Promise<T> {
  if (!workspaceId.trim()) {
    throw new Error("Plane workspace id is required");
  }
  const backend = getEffectiveLocalBackend();
  if (!backend) {
    throw new NoBackendAvailableError();
  }

  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const url = `${backend.host.replace(/\/+$/, "")}${PLANE_PROXY_BASE}${normalizedPath}`;
  const apiKey = backend.apiKey?.trim();

  const response = await axios.request<T>({
    method,
    url,
    headers: {
      Accept: "application/json",
      [PLANE_WORKSPACE_ID_HEADER]: workspaceId,
      ...(apiKey ? { "X-Session-API-Key": apiKey } : {}),
    },
  });
  return response.data;
}

/** Bound Plane client for a single local workspace. */
export type PlaneClient = ReturnType<typeof PlaneService.forWorkspace>;

export class PlaneService {
  static forWorkspace(workspaceId: string) {
    const id = workspaceId.trim();

    return {
      /**
       * Verifies credentials + project (and optional module) via the Canvas
       * proxy using the stored per-workspace Plane config.
       */
      async testConnection(): Promise<void> {
        await planeRequest(id, "GET", "/test");
      },
    };
  }
}
