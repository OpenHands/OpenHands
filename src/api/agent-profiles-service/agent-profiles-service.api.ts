/**
 * AgentProfilesService wraps the SDK's AgentProfilesClient, creating a client
 * per-call so it always picks up the active backend's host/apiKey (same pattern
 * as ProfilesService / SettingsService).
 *
 * Backs the Settings → Agent profiles library + reused agent editor. The
 * `AgentProfilesClient` ships in ts-client (pinned 1.28.0 here); the
 * `/api/agent-profiles` endpoints it targets shipped in agent-server v1.29.0
 * (local) and in the enterprise cloud app-server (OpenHands #15060, epic
 * #3730). Cloud exposes the SAME contract but authenticates with a bearer
 * token + `X-Org-Id`, so cloud calls route through `callCloudProxy` (see
 * `cloud/agent-profiles-service.api.ts`) instead of the direct client.
 */
import {
  AgentProfilesClient,
  type GetAgentProfileOptions,
} from "@openhands/typescript-client/clients";
import type {
  AgentProfile,
  AgentProfileSummary,
  AgentProfileSaveInput,
  AgentProfileListResponse,
  AgentProfileDetailResponse,
  AgentProfileMutationResponse,
  ActivateAgentProfileResponse,
  ExposeSecretsMode,
} from "@openhands/typescript-client";
import { getAgentServerClientOptions } from "../agent-server-client-options";
import { getActiveBackend } from "../backend-registry/active-store";
import {
  listCloudAgentProfiles,
  getCloudAgentProfile,
  saveCloudAgentProfile,
  deleteCloudAgentProfile,
  renameCloudAgentProfile,
  activateCloudAgentProfile,
} from "../cloud/agent-profiles-service.api";

function isCloud(): boolean {
  return getActiveBackend().backend.kind === "cloud";
}

/**
 * The seeded, well-known baseline agent profile. The backend lazily seeds it to
 * mirror the user's global config (#3719), and onboarding configures it from the
 * user's choice. It stands in for global `agent_settings`, so the home-launch
 * path treats it as the enriched baseline rather than a deliberate profile
 * selection (see `useCreateConversation`).
 */
export const WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME = "default";

// Re-export SDK types for consumers.
export type {
  AgentProfile,
  AgentProfileSummary,
  AgentProfileSaveInput,
  AgentProfileListResponse,
  AgentProfileDetailResponse,
  AgentProfileMutationResponse,
  ActivateAgentProfileResponse,
};

class AgentProfilesService {
  static async listProfiles(): Promise<AgentProfileListResponse> {
    if (isCloud()) return listCloudAgentProfiles();
    return new AgentProfilesClient(
      getAgentServerClientOptions(),
    ).listAgentProfiles();
  }

  static async getProfile(
    name: string,
    exposeSecrets?: ExposeSecretsMode,
  ): Promise<AgentProfileDetailResponse> {
    // Agent profiles are secret-free at rest now (embedded skills and their
    // `mcp_tools` were removed in #4017 — a profile carries only refs + the
    // `disabled_skills` deny-list of names), so `exposeSecrets` is vestigial and
    // cloud ignores it; kept for local signature parity with ProfilesService.
    if (isCloud()) return getCloudAgentProfile(name);
    const options: GetAgentProfileOptions = exposeSecrets
      ? { exposeSecrets }
      : {};
    return new AgentProfilesClient(
      getAgentServerClientOptions(),
    ).getAgentProfile(name, options);
  }

  /** Create or overwrite a profile by name (upsert). */
  static async saveProfile(
    name: string,
    profile: AgentProfileSaveInput,
  ): Promise<AgentProfileMutationResponse> {
    if (isCloud()) return saveCloudAgentProfile(name, profile);
    return new AgentProfilesClient(
      getAgentServerClientOptions(),
    ).saveAgentProfile(name, profile);
  }

  static async deleteProfile(
    name: string,
  ): Promise<AgentProfileMutationResponse> {
    if (isCloud()) return deleteCloudAgentProfile(name);
    return new AgentProfilesClient(
      getAgentServerClientOptions(),
    ).deleteAgentProfile(name);
  }

  static async renameProfile(
    name: string,
    newName: string,
  ): Promise<AgentProfileMutationResponse> {
    if (isCloud()) return renameCloudAgentProfile(name, newName);
    return new AgentProfilesClient(
      getAgentServerClientOptions(),
    ).renameAgentProfile(name, newName);
  }

  /** Activate by the profile's stable UUID `id` (pointer-only; never writes
   * agent_settings). */
  static async activateProfile(
    profileId: string,
  ): Promise<ActivateAgentProfileResponse> {
    if (isCloud()) return activateCloudAgentProfile(profileId);
    return new AgentProfilesClient(
      getAgentServerClientOptions(),
    ).activateAgentProfile(profileId);
  }
}

export default AgentProfilesService;
