import { SettingsClient } from "@openhands/typescript-client/clients";
import { isSdkHttpStatusError } from "./agent-server-compatibility";
import { getActiveBackend } from "./backend-registry/active-store";
import {
  createCloudSecret,
  deleteCloudSecret,
  fetchCloudSecrets,
  updateCloudSecret,
} from "./cloud/secrets-service.api";
import { getAgentServerClientOptions } from "./agent-server-client-options";
import { CustomSecretWithoutValue } from "./secrets-service.types";

/**
 * Retry helper for API calls with exponential backoff.
 */
async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  baseDelayMs: number = 500,
): Promise<T> {
  for (let attempt = 0; attempt < maxRetries; attempt += 1) {
    try {
      return await fn();
    } catch (error) {
      if (attempt >= maxRetries - 1) {
        throw error;
      }

      const delay = baseDelayMs * 2 ** attempt;

      await new Promise<void>((resolve) => {
        setTimeout(resolve, delay);
      });
    }
  }

  throw new Error("Retry attempts exhausted");
}

export class SecretsService {
  /**
   * List all custom secrets (names and descriptions only, no values).
   * Uses the agent-server API endpoint: GET /api/settings/secrets
   *
   * Note: The agent-server API doesn't support pagination or search filtering.
   * All secrets are returned in a single response.
   */
  static async getSecrets(): Promise<CustomSecretWithoutValue[]> {
    try {
      if (getActiveBackend().backend.kind === "cloud") {
        return await withRetry(() => fetchCloudSecrets());
      }
      const response = await withRetry(() =>
        new SettingsClient(getAgentServerClientOptions()).listSecrets(),
      );
      return response.secrets.map((s) => ({
        name: s.name,
        description: s.description,
      }));
    } catch (error) {
      console.error("Failed to fetch secrets after retries:", error);
      return [];
    }
  }

  /**
   * Create or update a custom secret (upsert by name).
   * Uses the agent-server API endpoint: PUT /api/settings/secrets
   *
   * @param name - Secret name (must start with letter, contain only letters/numbers/underscores, 1-64 chars)
   * @param value - Secret value
   * @param description - Optional description
   * @throws Error if the API call fails after retries
   */
  static async createSecret(
    name: string,
    value: string,
    description?: string,
  ): Promise<void> {
    if (getActiveBackend().backend.kind === "cloud") {
      await withRetry(() => createCloudSecret(name, value, description));
      return;
    }
    await withRetry(() =>
      new SettingsClient(getAgentServerClientOptions()).upsertSecret({
        name,
        value,
        description,
      }),
    );
  }

  /**
   * Update a secret's name and/or description, and optionally overwrite its
   * value. When no value is given the existing one is preserved: the
   * agent-server only exposes an upsert endpoint, so we fetch the existing
   * value and re-upsert it under the updated name/description.
   *
   * @param secretToEdit - Existing secret name
   * @param name - New (or same) secret name
   * @param description - Optional new description
   * @param value - Optional new value; when omitted the stored value is kept
   * @throws Error if the API call fails after retries
   */
  static async updateSecret(
    secretToEdit: string,
    name: string,
    description?: string,
    value?: string,
  ): Promise<void> {
    if (getActiveBackend().backend.kind === "cloud") {
      await withRetry(() => updateCloudSecret(secretToEdit, name, description));
      if (value !== undefined) {
        // The cloud PUT cannot change a value; POST is the only endpoint that
        // overwrites one. Running it after the PUT keeps the server-side
        // rename-collision check as the first thing that can fail.
        await withRetry(() => createCloudSecret(name, value, description));
      }
      return;
    }

    const client = new SettingsClient(getAgentServerClientOptions());
    const nextValue =
      value ?? (await withRetry(() => client.getSecret(secretToEdit)));

    await withRetry(() =>
      client.upsertSecret({
        name,
        value: nextValue,
        description,
      }),
    );

    if (name !== secretToEdit) {
      await this.deleteSecret(secretToEdit);
    }
  }

  /**
   * Delete a custom secret by name.
   * Uses the agent-server API endpoint: DELETE /api/settings/secrets/{name}
   *
   * @param name - Secret name to delete
   * @throws Error if the API call fails (except 404, which is treated as success)
   */
  static async deleteSecret(name: string): Promise<void> {
    try {
      if (getActiveBackend().backend.kind === "cloud") {
        await withRetry(() => deleteCloudSecret(name));
        return;
      }
      await withRetry(() =>
        new SettingsClient(getAgentServerClientOptions()).deleteSecret(name),
      );
    } catch (error) {
      // 404 means secret doesn't exist - treat as successful deletion.
      // Both the SDK's HttpError (status on the error itself) and
      // axios-style errors (status under `response`) count.
      if (
        isSdkHttpStatusError(error, 404) ||
        (error &&
          typeof error === "object" &&
          "response" in error &&
          (error as { response?: { status?: number } }).response?.status ===
            404)
      ) {
        return;
      }
      throw error;
    }
  }
}
