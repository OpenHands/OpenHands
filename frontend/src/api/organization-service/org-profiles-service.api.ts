import { openHands } from "../open-hands-axios";

export interface OrgLlmProfileSummary {
  name: string;
  model: string | null;
  base_url: string | null;
  api_key_set: boolean;
}

interface OrgLlmProfileListResponse {
  profiles: OrgLlmProfileSummary[];
  active_profile: string | null;
}

export interface SaveOrgLlmProfileRequest {
  include_secrets?: boolean;
  // True when the user typed no new key, so the backend keeps the profile's
  // stored key instead of snapshotting the active settings' key.
  preserve_existing_api_key?: boolean;
  llm?: {
    model: string;
    base_url?: string | null;
    api_key?: string | null;
  } & Record<string, unknown>;
}

class OrgProfilesService {
  static async listProfiles(orgId: string): Promise<OrgLlmProfileListResponse> {
    const { data } = await openHands.get<OrgLlmProfileListResponse>(
      `/api/organizations/${orgId}/profiles`,
    );
    return data;
  }

  static async getProfile(
    orgId: string,
    name: string,
  ): Promise<{ name: string; llm: Record<string, unknown> }> {
    const { data } = await openHands.get(
      `/api/organizations/${orgId}/profiles/${encodeURIComponent(name)}`,
    );
    return data;
  }

  static async saveProfile(
    orgId: string,
    name: string,
    request: SaveOrgLlmProfileRequest = {},
  ): Promise<void> {
    await openHands.post(
      `/api/organizations/${orgId}/profiles/${encodeURIComponent(name)}`,
      request,
    );
  }

  static async deleteProfile(orgId: string, name: string): Promise<void> {
    await openHands.delete(
      `/api/organizations/${orgId}/profiles/${encodeURIComponent(name)}`,
    );
  }

  static async activateProfile(orgId: string, name: string): Promise<void> {
    await openHands.post(
      `/api/organizations/${orgId}/profiles/${encodeURIComponent(name)}/activate`,
    );
  }

  static async renameProfile(
    orgId: string,
    name: string,
    newName: string,
  ): Promise<void> {
    await openHands.post(
      `/api/organizations/${orgId}/profiles/${encodeURIComponent(name)}/rename`,
      { new_name: newName },
    );
  }
}

export default OrgProfilesService;
