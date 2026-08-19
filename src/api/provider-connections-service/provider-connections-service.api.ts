/**
 * ProviderConnectionsService is a thin wrapper over the local agent-server's
 * `/api/llm/provider-connections` CRUD endpoints. A provider connection is a
 * shared `api_key` + optional `base_url` that one or more LLM profiles
 * reference by id; the agent-server resolves the credential into a runnable LLM
 * at profile-load time, so this service only manages the stored connections.
 *
 * These endpoints exist only on the agent-server (local backend). Cloud has no
 * equivalent yet, so callers must gate usage on `backend.kind === "local"`.
 *
 * There is no generated client for these routes in `@openhands/typescript-client`
 * yet, so requests go through the generic `AgentServerClient` verb helpers —
 * the same approach `LLMBalanceService` uses.
 */
import { AgentServerClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "../agent-server-client-options";

const PROVIDER_CONNECTIONS_PATH = "/api/llm/provider-connections";

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
