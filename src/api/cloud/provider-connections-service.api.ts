import type {
  ProviderConnection,
  CreateProviderConnectionRequest,
  UpdateProviderConnectionRequest,
} from "../provider-connections-service/provider-connections-service.api";
import { getActiveBackend } from "../backend-registry/active-store";
import type { Backend } from "../backend-registry/types";
import { callCloudProxy } from "./proxy";

/**
 * Cloud provider-connections service.
 *
 * Provider-connection CRUD is routed to the org-scoped endpoints
 * `/api/organizations/{orgId}/provider-connections`, which enforce
 * `EDIT_ORG_SETTINGS` server-side (listing requires `VIEW_ORG_SETTINGS`) — so a
 * member's mutation is rejected with 403 even on a direct API call, not just
 * hidden in the UI. This mirrors the cloud LLM-profile service.
 *
 * Unlike the local agent-server (which returns a bare array), the org list
 * route wraps results as `{ connections: [...] }`; this service unwraps that so
 * callers see the same `ProviderConnection[]` shape on both backends. Individual
 * mutations return a bare `ProviderConnection` on both backends. Neither route
 * exposes the stored key; each item carries `api_key_set` instead.
 *
 * Cloud has no unbound-org fallback here: provider connections are inherently
 * org-scoped, so an org must be bound. A cloud backend without an org id is a
 * programming error rather than a legacy-key case.
 */

interface CloudProviderConnectionListResponse {
  connections: ProviderConnection[];
}

/** Resolve the backend + org-scoped base path for the active cloud backend. */
function cloudProviderConnectionsTarget(): { backend: Backend; base: string } {
  const { backend, orgId } = getActiveBackend();
  if (backend.kind !== "cloud") {
    throw new Error(
      "Cloud provider-connections call requires a cloud backend.",
    );
  }
  if (!orgId) {
    throw new Error(
      "Cloud provider connections require an organization-bound backend.",
    );
  }
  return {
    backend,
    base: `/api/organizations/${encodeURIComponent(orgId)}/provider-connections`,
  };
}

export async function fetchCloudProviderConnections(): Promise<
  ProviderConnection[]
> {
  const { backend, base } = cloudProviderConnectionsTarget();
  const result = await callCloudProxy<CloudProviderConnectionListResponse>({
    backend,
    method: "GET",
    path: base,
  });
  return result.connections ?? [];
}

export async function createCloudProviderConnection(
  request: CreateProviderConnectionRequest,
): Promise<ProviderConnection> {
  const { backend, base } = cloudProviderConnectionsTarget();
  return callCloudProxy<ProviderConnection>({
    backend,
    method: "POST",
    path: base,
    body: request,
  });
}

export async function updateCloudProviderConnection(
  id: string,
  request: UpdateProviderConnectionRequest,
): Promise<ProviderConnection> {
  const { backend, base } = cloudProviderConnectionsTarget();
  return callCloudProxy<ProviderConnection>({
    backend,
    method: "PATCH",
    path: `${base}/${encodeURIComponent(id)}`,
    body: request,
  });
}

export async function deleteCloudProviderConnection(
  id: string,
): Promise<ProviderConnection> {
  const { backend, base } = cloudProviderConnectionsTarget();
  return callCloudProxy<ProviderConnection>({
    backend,
    method: "DELETE",
    path: `${base}/${encodeURIComponent(id)}`,
  });
}
