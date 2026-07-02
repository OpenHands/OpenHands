import { openHands } from "./open-hands-axios";

export interface ApiKey {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  last_used_at: string | null;
  not_before: string | null;
  expires_at: string | null;
  /**
   * Org the key is bound to. ``null`` means the key is *unbound* and can be
   * scoped per-request via the ``X-Org-Id`` header (falling back to the
   * user's current org when the header is absent).
   */
  org_id: string | null;
}

export interface CreateApiKeyResponse {
  id: string;
  name: string;
  key: string; // Full key, only returned once upon creation
  prefix: string;
  created_at: string;
  not_before: string | null;
  expires_at: string | null;
  org_id: string | null;
}

export interface CreateApiKeyInput {
  name: string;
  not_before?: string | null; // ISO 8601 UTC; omit to activate immediately
  expires_at?: string | null; // ISO 8601 UTC; omit for no expiration
  /**
   * Org to bind the new key to.
   * - ``null`` (explicit): unbound key -- usable against any org via
   *   ``X-Org-Id`` or the caller's current org.
   * - ``undefined`` (omitted): bind to the request's effective org.
   * - UUID string: bind to the specified org (caller must be a member).
   */
  org_id?: string | null;
}

class ApiKeysClient {
  /**
   * Get all API keys for the current user.
   *
   * Returns keys that are either bound to the effective org or unbound
   * (``org_id: null`` -- visible regardless of the active org context).
   */
  static async getApiKeys(): Promise<ApiKey[]> {
    const { data } = await openHands.get<unknown>("/api/keys");
    // Ensure we always return an array, even if the API returns something else
    return Array.isArray(data) ? (data as ApiKey[]) : [];
  }

  /**
   * Create a new API key
   * @param input - Key name, optional active-window bounds, and optional
   *   org binding (see ``CreateApiKeyInput.org_id``).
   */
  static async createApiKey(
    input: CreateApiKeyInput,
  ): Promise<CreateApiKeyResponse> {
    // Forward the raw ``org_id`` field as-is so that an explicit ``null``
    // ("All orgs" / unbound key) is preserved through the JSON request.
    const body: Record<string, unknown> = {
      name: input.name,
      not_before: input.not_before ?? undefined,
      expires_at: input.expires_at ?? undefined,
    };
    if ("org_id" in input) {
      body.org_id = input.org_id;
    }
    const { data } = await openHands.post<CreateApiKeyResponse>(
      "/api/keys",
      body,
    );
    return data;
  }

  /**
   * Delete an API key
   * @param id - The ID of the API key to delete
   */
  static async deleteApiKey(id: string): Promise<void> {
    await openHands.delete(`/api/keys/${id}`);
  }
}

export default ApiKeysClient;
