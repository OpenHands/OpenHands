import { openHands } from "../open-hands-axios";

export interface LlmProfileSummary {
  name: string;
  model: string | null;
  base_url: string | null;
  api_key_set: boolean;
}

// Not exported — only `listProfiles` reads it as its response shape.
interface LlmProfileListResponse {
  profiles: LlmProfileSummary[];
  active_profile: string | null;
}

export interface SaveLlmProfileRequest {
  include_secrets?: boolean;
  llm?: {
    model: string;
    base_url?: string | null;
    api_key?: string | null;
  } & Record<string, unknown>;
}

class ProfilesService {
  static async listProfiles(): Promise<LlmProfileListResponse> {
    const { data } = await openHands.get<LlmProfileListResponse>(
      "/api/v1/settings/profiles",
    );
    return data;
  }

  static async saveProfile(
    name: string,
    request: SaveLlmProfileRequest = {},
  ): Promise<void> {
    await openHands.post(
      `/api/v1/settings/profiles/${encodeURIComponent(name)}`,
      request,
    );
  }

  static async deleteProfile(name: string): Promise<void> {
    await openHands.delete(
      `/api/v1/settings/profiles/${encodeURIComponent(name)}`,
    );
  }

  static async activateProfile(name: string): Promise<void> {
    await openHands.post(
      `/api/v1/settings/profiles/${encodeURIComponent(name)}/activate`,
    );
  }

  static async renameProfile(name: string, newName: string): Promise<void> {
    await openHands.post(
      `/api/v1/settings/profiles/${encodeURIComponent(name)}/rename`,
      { new_name: newName },
    );
  }
}

export default ProfilesService;
