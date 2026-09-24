/**
 * ProviderConnectionsService is the single entry point for provider-connection
 * CRUD, routing per active backend so callers (hooks, the settings manager)
 * stay backend-agnostic. A provider connection is a shared `api_key` + optional
 * `base_url` that one or more LLM profiles reference by id; the backend resolves
 * the credential into a runnable LLM at profile-load time, so this service only
 * manages the stored connections.
 *
 * - local agent-server: the `/api/llm/provider-connections` CRUD endpoints via
 *   the generic `AgentServerClient` verb helpers (the same approach
 *   `LLMBalanceService` uses; there is no generated client for these routes in
 *   `@openhands/typescript-client` yet);
 * - cloud app-server: `src/api/cloud/provider-connections-service.api.ts` (the
 *   org-gated `/api/organizations/{orgId}/provider-connections` routes) via the
 *   org-scoped cloud proxy.
 *
 * This mirrors how `ProfilesService` branches to the cloud profile service.
 */
import { AgentServerClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "../agent-server-client-options";
import { getActiveBackend } from "../backend-registry/active-store";
import {
  fetchCloudProviderConnections,
  createCloudProviderConnection,
  updateCloudProviderConnection,
  deleteCloudProviderConnection,
} from "../cloud/provider-connections-service.api";

const PROVIDER_CONNECTIONS_PATH = "/api/llm/provider-connections";

function isCloudBackend(): boolean {
  return getActiveBackend().backend.kind === "cloud";
}

export interface ProviderConnection {
  id: string;
  display_name: string;
  provider: string;
  base_url: string | null;
  created_at: number;
  updated_at: number;
  /** Whether the stored connection currently holds a usable key. */
  api_key_set: boolean;
}

export interface CreateProviderConnectionRequest {
  display_name: string;
  provider: string;
  api_key: string;
  base_url?: string | null;
}

/**
 * Partial update. Only the provided fields change. `api_key` may be sent to
 * rotate the key; the agent-server rejects `api_key: null` (a connection must
 * always keep a key), so callers omit it to leave the key unchanged.
 */
export interface UpdateProviderConnectionRequest {
  display_name?: string;
  provider?: string;
  api_key?: string;
  base_url?: string | null;
}

function createClient(): AgentServerClient {
  const { host, apiKey } = getAgentServerClientOptions();
  return new AgentServerClient({ host, ...(apiKey ? { apiKey } : {}) });
}

class ProviderConnectionsService {
  static async list(): Promise<ProviderConnection[]> {
    if (isCloudBackend()) return fetchCloudProviderConnections();
    const client = createClient();
    try {
      return await client.get<ProviderConnection[]>(PROVIDER_CONNECTIONS_PATH, {
        responseType: "json",
      });
    } finally {
      client.close();
    }
  }

  static async create(
    request: CreateProviderConnectionRequest,
  ): Promise<ProviderConnection> {
    if (isCloudBackend()) return createCloudProviderConnection(request);
    const client = createClient();
    try {
      return await client.post<ProviderConnection>(
        PROVIDER_CONNECTIONS_PATH,
        request,
        { responseType: "json" },
      );
    } finally {
      client.close();
    }
  }

  static async update(
    id: string,
    request: UpdateProviderConnectionRequest,
  ): Promise<ProviderConnection> {
    if (isCloudBackend()) return updateCloudProviderConnection(id, request);
    const client = createClient();
    try {
      return await client.patch<ProviderConnection>(
        `${PROVIDER_CONNECTIONS_PATH}/${encodeURIComponent(id)}`,
        request,
        { responseType: "json" },
      );
    } finally {
      client.close();
    }
  }

  static async delete(id: string): Promise<ProviderConnection> {
    if (isCloudBackend()) return deleteCloudProviderConnection(id);
    const client = createClient();
    try {
      return await client.delete<ProviderConnection>(
        `${PROVIDER_CONNECTIONS_PATH}/${encodeURIComponent(id)}`,
        { responseType: "json" },
      );
    } finally {
      client.close();
    }
  }
}

export default ProviderConnectionsService;
