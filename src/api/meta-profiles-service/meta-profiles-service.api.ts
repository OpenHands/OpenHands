/**
 * MetaProfilesService wraps the agent-server's ``/api/meta-profiles`` endpoints
 * (added in software-agent-sdk PR #3744). A meta-profile is a model-routing
 * configuration consumed by the ``classify_and_switch_llm`` tool: it names a
 * ``classifier_model``, a ``default_model`` and a list of task ``classes``,
 * where every model reference is the name of a saved LLM profile.
 *
 * The SDK's ``@openhands/typescript-client`` does not (yet) ship a dedicated
 * meta-profiles client, so we drive the endpoints with the SDK's public
 * ``HttpClient`` — mirroring how ``ProfilesClient`` is implemented — and create
 * a client per call to pick up the current backend configuration.
 */
import { HttpClient } from "@openhands/typescript-client";
import { getAgentServerClientOptions } from "../agent-server-client-options";

export interface MetaProfileClass {
  description: string;
  /** Name of the saved LLM profile to switch to for this class. */
  model: string;
}

export interface MetaProfile {
  /** Name of the saved LLM profile used to classify the task. */
  classifier_model: string;
  /** Name of the saved LLM profile to use when no class matches. */
  default_model: string;
  classes: MetaProfileClass[];
}

export interface MetaProfileInfo {
  name: string;
  classifier_model: string | null;
  default_model: string | null;
  num_classes: number;
}

export interface MetaProfileListResponse {
  meta_profiles: MetaProfileInfo[];
  active_meta_profile: string | null;
}

export interface MetaProfileDetailResponse {
  name: string;
  config: MetaProfile;
}

export interface MetaProfileMutationResponse {
  name: string;
  message: string;
}

export interface ActivateMetaProfileResponse {
  name: string;
  message: string;
}

const BASE_PATH = "/api/meta-profiles";

function client(): HttpClient {
  const { host, apiKey } = getAgentServerClientOptions();
  return new HttpClient({ baseUrl: host, apiKey, timeout: 60000 });
}

class MetaProfilesService {
  static async listMetaProfiles(): Promise<MetaProfileListResponse> {
    const response = await client().get<MetaProfileListResponse>(BASE_PATH);
    return response.data;
  }

  static async getMetaProfile(
    name: string,
  ): Promise<MetaProfileDetailResponse> {
    const response = await client().get<MetaProfileDetailResponse>(
      `${BASE_PATH}/${encodeURIComponent(name)}`,
    );
    return response.data;
  }

  static async saveMetaProfile(
    name: string,
    config: MetaProfile,
  ): Promise<MetaProfileMutationResponse> {
    const response = await client().post<MetaProfileMutationResponse>(
      `${BASE_PATH}/${encodeURIComponent(name)}`,
      config,
    );
    return response.data;
  }

  static async deleteMetaProfile(
    name: string,
  ): Promise<MetaProfileMutationResponse> {
    const response = await client().delete<MetaProfileMutationResponse>(
      `${BASE_PATH}/${encodeURIComponent(name)}`,
    );
    return response.data;
  }

  static async activateMetaProfile(
    name: string,
  ): Promise<ActivateMetaProfileResponse> {
    const response = await client().post<ActivateMetaProfileResponse>(
      `${BASE_PATH}/${encodeURIComponent(name)}/activate`,
      {},
    );
    return response.data;
  }
}

export default MetaProfilesService;
