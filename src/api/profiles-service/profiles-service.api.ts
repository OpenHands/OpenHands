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
  ProfileInfo,
  ProfileListResponse,
  ProfileDetailResponse,
  ProfileMutationResponse,
  ActivateProfileResponse,
  SaveProfileRequest,
  ExposeSecretsMode,
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

// Re-export SDK types for consumers
export type {
  ProfileInfo,
  ProfileListResponse,
  ProfileDetailResponse,
  ProfileMutationResponse,
  ActivateProfileResponse,
  SaveProfileRequest,
  ExposeSecretsMode,
};

function isCloudBackend(): boolean {
  return getActiveBackend().backend.kind === "cloud";
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
}

export default ProfilesService;
