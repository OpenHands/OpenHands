import { openHands } from "./open-hands-axios";

export interface ApiKey {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  last_used_at: string | null;
  not_before: string | null;
  expires_at: string | null;
}

export interface CreateApiKeyResponse {
  id: string;
  name: string;
  key: string; // Full key, only returned once upon creation
  prefix: string;
  created_at: string;
  not_before: string | null;
  expires_at: string | null;
}

export interface CreateApiKeyInput {
  name: string;
  not_before?: string | null; // ISO 8601 UTC; omit to activate immediately
  expires_at?: string | null; // ISO 8601 UTC; omit for no expiration
}

class ApiKeysClient {
  /**
   * Get all API keys for the current user
   */
  static async getApiKeys(): Promise<ApiKey[]> {
    const { data } = await openHands.get<unknown>("/api/keys");
    // Ensure we always return an array, even if the API returns something else
    return Array.isArray(data) ? (data as ApiKey[]) : [];
  }

  /**
   * Create a new API key
   * @param input - Key name plus optional active-window bounds
   */
  static async createApiKey(
    input: CreateApiKeyInput,
  ): Promise<CreateApiKeyResponse> {
    const { data } = await openHands.post<CreateApiKeyResponse>("/api/keys", {
      name: input.name,
      not_before: input.not_before ?? undefined,
      expires_at: input.expires_at ?? undefined,
    });
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
