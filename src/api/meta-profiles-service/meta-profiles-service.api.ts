/**
 * MetaProfilesService wraps the agent-server's ``/api/meta-profiles`` endpoints
 * (added in software-agent-sdk PR #4287). A meta-profile is a model-routing
 * configuration consumed by the ``route_task_to_model`` tool: it names a
 * ``classifier_model`` and a direct ``prompt_template``
 * whose returned ``model`` value is matched to saved LLM profile names.
 *
 * Transport goes through the SDK's typed ``MetaProfilesClient`` (mirroring how
 * ``ProfilesService`` uses ``ProfilesClient``), creating a client per call so
 * it picks up the current backend configuration.
 *
 * Types are re-exported from the SDK for consumer convenience.
 */
import { MetaProfilesClient } from "@openhands/typescript-client/clients";
import type {
  ActivateMetaProfileResponse,
  MetaProfile,
  MetaProfileClass,
  MetaProfileDetailResponse,
  MetaProfileInfo,
  MetaProfileListResponse,
  MetaProfileMutationResponse,
} from "@openhands/typescript-client";
import { getAgentServerClientOptions } from "../agent-server-client-options";
import { getActiveBackend } from "../backend-registry/active-store";
import {
  activateCloudMetaProfile,
  deleteCloudMetaProfile,
  fetchCloudMetaProfile,
  fetchCloudMetaProfiles,
  saveCloudMetaProfile,
} from "../cloud/meta-profiles-service.api";

// Re-export SDK types for consumers
export type {
  ActivateMetaProfileResponse,
  MetaProfile,
  MetaProfileClass,
  MetaProfileDetailResponse,
  MetaProfileInfo,
  MetaProfileListResponse,
  MetaProfileMutationResponse,
};

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
    return new MetaProfilesClient(getAgentServerClientOptions()).getMetaProfile(
      name,
    );
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
    ).saveMetaProfile(name, config);
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
