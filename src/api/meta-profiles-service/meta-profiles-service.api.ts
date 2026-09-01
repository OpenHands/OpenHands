/**
 * MetaProfilesService wraps the agent-server's ``/api/meta-profiles`` endpoints
 * (added in software-agent-sdk PR #4287). A meta-profile is a model-routing
 * configuration consumed by the ``classify_and_switch_llm`` tool: it names a
 * ``classifier_model``, a ``default_model`` and a direct ``prompt_template``
 * whose returned ``model`` value is matched to saved LLM profile names.
 *
 * Transport goes through the SDK's typed ``MetaProfilesClient`` (mirroring how
 * ``ProfilesService`` uses ``ProfilesClient``), creating a client per call so
 * it picks up the current backend configuration.
 *
 * The local ``MetaProfile`` type is intentionally wider than the SDK's: this
 * UI writes direct-prompt profiles, so it carries ``prompt_template`` /
 * ``model_table`` (and treats ``classes`` as optional) which the SDK's
 * ``MetaProfile`` type does not declare yet. ``MetaProfilesClient`` forwards
 * the config to the server unchanged, so we cast the local config to the SDK
 * type on save and the extra fields still reach the backend. Once the SDK type
 * is widened, the cast can be dropped.
 */
import { MetaProfilesClient } from "@openhands/typescript-client/clients";
import type { MetaProfile as SdkMetaProfile } from "@openhands/typescript-client";
import { getAgentServerClientOptions } from "../agent-server-client-options";
import { getActiveBackend } from "../backend-registry/active-store";
import {
  activateCloudMetaProfile,
  deleteCloudMetaProfile,
  fetchCloudMetaProfile,
  fetchCloudMetaProfiles,
  saveCloudMetaProfile,
} from "../cloud/meta-profiles-service.api";

export interface MetaProfileClass {
  description: string;
  /** Name of the saved LLM profile to switch to for this class. */
  model: string;
}

export interface MetaProfile {
  /** Name of the saved LLM profile used to classify the task. */
  classifier_model: string;
  /** Name of the saved LLM profile to use when routing fails or no label matches. */
  default_model: string;
  /** Structured classes are kept for backend compatibility, but this UI writes direct prompt profiles. */
  classes?: MetaProfileClass[];
  /** Direct-routing prompt template. Must include ``{{ instance_text }}`` when set. */
  prompt_template: string | null;
  /** Optional text inserted into ``{{ model_table }}`` by the backend. */
  model_table: string | null;
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

class MetaProfilesService {
  static async listMetaProfiles(): Promise<MetaProfileListResponse> {
    if (getActiveBackend().backend.kind === "cloud") {
      return fetchCloudMetaProfiles();
    }
    return new MetaProfilesClient(
      getAgentServerClientOptions(),
    ).listMetaProfiles();
  }

  static async getMetaProfile(
    name: string,
  ): Promise<MetaProfileDetailResponse> {
    if (getActiveBackend().backend.kind === "cloud") {
      return fetchCloudMetaProfile(name);
    }
    // The server returns the full direct-prompt config (including
    // ``prompt_template`` / ``model_table``); the SDK's ``MetaProfile`` type
    // just does not declare those fields yet, so widen the typed result.
    return new MetaProfilesClient(getAgentServerClientOptions()).getMetaProfile(
      name,
    ) as unknown as Promise<MetaProfileDetailResponse>;
  }

  static async saveMetaProfile(
    name: string,
    config: MetaProfile,
  ): Promise<MetaProfileMutationResponse> {
    if (getActiveBackend().backend.kind === "cloud") {
      return saveCloudMetaProfile(name, config);
    }
    return new MetaProfilesClient(
      getAgentServerClientOptions(),
    ).saveMetaProfile(name, config as unknown as SdkMetaProfile);
  }

  static async deleteMetaProfile(
    name: string,
  ): Promise<MetaProfileMutationResponse> {
    if (getActiveBackend().backend.kind === "cloud") {
      return deleteCloudMetaProfile(name);
    }
    return new MetaProfilesClient(
      getAgentServerClientOptions(),
    ).deleteMetaProfile(name);
  }

  static async activateMetaProfile(
    name: string,
  ): Promise<ActivateMetaProfileResponse> {
    if (getActiveBackend().backend.kind === "cloud") {
      return activateCloudMetaProfile(name);
    }
    return new MetaProfilesClient(
      getAgentServerClientOptions(),
    ).activateMetaProfile(name);
  }
}

export default MetaProfilesService;
