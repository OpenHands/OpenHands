import { CloudClient } from "@openhands/typescript-client/clients";
import {
  getAgentServerBaseUrl,
  getAgentServerHeaders,
} from "../agent-server-config";
import { NoBackendAvailableError } from "../agent-server-client-options";
import { getActiveBackend } from "../backend-registry/active-store";
import type { Backend } from "../backend-registry/types";

function requireCloudBackend(backend?: Backend): Backend {
  if (backend) return backend;
  const active = getActiveBackend().backend;
  if (active.kind !== "cloud") {
    throw new Error("Cloud calls require a cloud backend.");
  }
  return active;
}

/**
 * Send `X-Org-Id` so the upstream scopes per-request to the org the user
 * selected locally, instead of the user's globally-shared `current_org_id`
 * on the cloud backend. Restricted to calls against the active backend: the
 * selector also fans out per-backend bookkeeping calls (e.g.
 * `getCloudOrganizations(b)`) that would otherwise carry the active
 * backend's orgId across an unrelated API key, which the cloud backend
 * rejects when api_key_org_id and X-Org-Id disagree.
 */
function activeOrgForBackend(backend: Backend): string | null {
  const active = getActiveBackend();
  return active.backend.id === backend.id ? active.orgId : null;
}

export function createCloudClient(backend?: Backend): CloudClient {
  const target = requireCloudBackend(backend);
  const agentServerProxyBaseUrl = getAgentServerBaseUrl();
  // Standalone Canvas uses its local Agent Server as the runtime proxy. The
  // SaaS Canvas is frontend-only, so fall back to the selected Cloud host,
  // which exposes the same /api/cloud-proxy endpoint.
  const proxyBaseUrl = agentServerProxyBaseUrl ?? target.host;
  const proxyHeaders = agentServerProxyBaseUrl ? getAgentServerHeaders() : {};

  return new CloudClient({
    host: target.host,
    apiKey: target.apiKey,
    orgId: activeOrgForBackend(target),
    // Default request timeout in ms, matching the 30s the previous axios
    // transport used for direct and proxied calls. Per-request
    // `timeoutSeconds` still overrides it for direct calls.
    timeout: 30_000,
    proxy: {
      host: proxyBaseUrl,
      headers: proxyHeaders,
    },
  });
}

export function createCloudClientForRuntime(backend?: Backend): CloudClient {
  const client = createCloudClient(backend);
  if (!client.proxy) {
    throw new NoBackendAvailableError();
  }
  return client;
}
