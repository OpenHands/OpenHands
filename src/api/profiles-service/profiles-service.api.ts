/**
 * ProfilesService is the single entry point for LLM-profile CRUD, routing per
 * active backend so callers (hooks, the settings manager) stay backend-agnostic:
 * - local agent-server: the SDK's ProfilesClient (`/api/profiles`), created
 *   per-call to pick up current backend configuration;
 * - cloud app-server: `src/api/cloud/profiles-service.api.ts` (the org-gated
 *   `/api/organizations/{orgId}/profiles` routes, or the per-user settings
 *   route as a fallback) via the org-scoped cloud proxy.
 * This mirrors how SettingsService branches to fetchCloudSettings().
 *
 * Uses ProfilesClient from @openhands/typescript-client v0.2.0+.
 * All types are re-exported from the SDK for consumer convenience.
 *
 * Note: Unlike some SDK clients, we don't call client.close() here for
 * consistency with other services (SettingsService, SecretsService) that
 * also create SDK clients without explicit cleanup. The SDK clients use
 * fetch-based HTTP which doesn't require connection cleanup.
 */
import {
  ProfilesClient,
  type GetProfileOptions,
} from "@openhands/typescript-client/clients";
import type {
  ProfileInfo as ClientProfileInfo,
  ProfileListResponse as ClientProfileListResponse,
  ProfileDetailResponse,
  ProfileMutationResponse,
  ActivateProfileResponse,
  SaveProfileRequest,
  ExposeSecretsMode,
  ValidateProfileResponse,
} from "@openhands/typescript-client";
import { getAgentServerClientOptions } from "../agent-server-client-options";
import { getActiveBackend } from "../backend-registry/active-store";
import {
  activateCloudProfile,
  deleteCloudProfile,
  fetchCloudProfile,
  fetchCloudProfiles,
  renameCloudProfile,
  saveCloudProfile,
} from "../cloud/profiles-service.api";

/**
 * Profile summaries carry an optional `provider_connection_id` (the shared
 * provider connection a profile links to), but `@openhands/typescript-client`
 * predates that field. Widen the client types here so consumers can read it; it
 * stays optional, so a client response without the field is still assignable.
 */
export interface ProfileInfo extends ClientProfileInfo {
  provider_connection_id?: string | null;
  /** True when provider_connection_id is set but the referenced connection no longer exists. */
  provider_connection_broken?: boolean;
}

export interface ProfileListResponse extends Omit<
  ClientProfileListResponse,
  "profiles"
> {
  profiles: ProfileInfo[];
}

// Re-export SDK types for consumers
export type {
  ProfileDetailResponse,
  ProfileMutationResponse,
  ActivateProfileResponse,
  SaveProfileRequest,
  ExposeSecretsMode,
  ValidateProfileResponse,
};

function isCloudBackend(): boolean {
  return getActiveBackend().backend.kind === "cloud";
}

function isAbortLike(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  const name = "name" in error ? error.name : undefined;
  if (name === "AbortError" || name === "TimeoutError") return true;
  const cause = "cause" in error ? error.cause : undefined;
  return (
    !!cause &&
    typeof cause === "object" &&
    "name" in cause &&
    (cause.name === "AbortError" || cause.name === "TimeoutError")
  );
}

class ProfilesService {
  static async listProfiles(): Promise<ProfileListResponse> {
    if (isCloudBackend()) return fetchCloudProfiles();
    return new ProfilesClient(getAgentServerClientOptions()).listProfiles();
  }

  static async getProfile(
    name: string,
    exposeSecrets?: ExposeSecretsMode,
  ): Promise<ProfileDetailResponse> {
    // Cloud never exposes profile secrets (api_key is always nulled with an
    // api_key_set flag), so `exposeSecrets` is local-only.
    if (isCloudBackend()) return fetchCloudProfile(name);
    const options: GetProfileOptions = exposeSecrets ? { exposeSecrets } : {};
    return new ProfilesClient(getAgentServerClientOptions()).getProfile(
      name,
      options,
    );
  }

  static async saveProfile(
    name: string,
    request: SaveProfileRequest,
  ): Promise<ProfileMutationResponse> {
    if (isCloudBackend()) return saveCloudProfile(name, request);
    return new ProfilesClient(getAgentServerClientOptions()).saveProfile(
      name,
      request,
    );
  }

  static async deleteProfile(name: string): Promise<ProfileMutationResponse> {
    if (isCloudBackend()) return deleteCloudProfile(name);
    return new ProfilesClient(getAgentServerClientOptions()).deleteProfile(
      name,
    );
  }

  static async renameProfile(
    name: string,
    newName: string,
  ): Promise<ProfileMutationResponse> {
    if (isCloudBackend()) return renameCloudProfile(name, newName);
    return new ProfilesClient(getAgentServerClientOptions()).renameProfile(
      name,
      newName,
    );
  }

  static async activateProfile(name: string): Promise<ActivateProfileResponse> {
    if (isCloudBackend()) return activateCloudProfile(name);
    return new ProfilesClient(getAgentServerClientOptions()).activateProfile(
      name,
    );
  }

  /**
   * Pre-flight check: fire a minimal LLM completion to catch misconfigurations
   * (invalid model names, missing provider prefixes, bad base URLs, invalid
   * API keys) before a profile is saved.
   *
   * Returns `{ valid: true }` when the LLM responds, or
   * `{ valid: false, error: { type, message } }` on a blocking error.
   * Transient errors (rate limits, timeouts) are non-blocking.
   *
   * Cloud backends do not implement this endpoint; `null` signals
   * "no verdict" so callers fall through to the normal save path.
   */
  static async validateProfile(
    name: string,
    request: SaveProfileRequest,
  ): Promise<ValidateProfileResponse | null> {
    if (isCloudBackend()) return null;
    const client = new ProfilesClient({
      ...getAgentServerClientOptions(),
      timeout: 30000,
    });
    try {
      return await client.validateProfile(name, request);
    } catch (error) {
      // Older agent-server versions don't have the endpoint → 404
      // Treat as "no verdict" rather than blocking the save.
      const status =
        error && typeof error === "object" && "status" in error
          ? (error as { status?: unknown }).status
          : undefined;
      if (
        status === 404 ||
        status === 429 ||
        (typeof status === "number" && status >= 500) ||
        isAbortLike(error)
      ) {
        return null;
      }
      throw error;
    } finally {
      client.close();
    }
  }
}

export default ProfilesService;
