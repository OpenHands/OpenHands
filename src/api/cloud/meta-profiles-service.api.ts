import { getActiveBackend } from "../backend-registry/active-store";
import type { Backend } from "../backend-registry/types";
import type {
  ActivateMetaProfileResponse,
  MetaProfile,
  MetaProfileDetailResponse,
  MetaProfileListResponse,
  MetaProfileMutationResponse,
} from "../meta-profiles-service/meta-profiles-service.api";
import { callCloudProxy } from "./proxy";

function cloudMetaProfilesTarget(): { backend: Backend; base: string } {
  const { backend, orgId } = getActiveBackend();
  if (backend.kind !== "cloud" || !orgId) {
    throw new Error(
      "Cloud Model Router settings require an organization-bound backend.",
    );
  }
  return {
    backend,
    base: `/api/organizations/${encodeURIComponent(orgId)}/meta-profiles`,
  };
}

export async function fetchCloudMetaProfiles(): Promise<MetaProfileListResponse> {
  const { backend, base } = cloudMetaProfilesTarget();
  return callCloudProxy<MetaProfileListResponse>({
    backend,
    method: "GET",
    path: base,
  });
}

export async function fetchCloudMetaProfile(
  name: string,
): Promise<MetaProfileDetailResponse> {
  const { backend, base } = cloudMetaProfilesTarget();
  return callCloudProxy<MetaProfileDetailResponse>({
    backend,
    method: "GET",
    path: `${base}/${encodeURIComponent(name)}`,
  });
}

export async function saveCloudMetaProfile(
  name: string,
  config: MetaProfile,
): Promise<MetaProfileMutationResponse> {
  const { backend, base } = cloudMetaProfilesTarget();
  return callCloudProxy<MetaProfileMutationResponse>({
    backend,
    method: "POST",
    path: `${base}/${encodeURIComponent(name)}`,
    body: config,
  });
}

export async function deleteCloudMetaProfile(
  name: string,
): Promise<MetaProfileMutationResponse> {
  const { backend, base } = cloudMetaProfilesTarget();
  return callCloudProxy<MetaProfileMutationResponse>({
    backend,
    method: "DELETE",
    path: `${base}/${encodeURIComponent(name)}`,
  });
}

export async function activateCloudMetaProfile(
  name: string,
): Promise<ActivateMetaProfileResponse> {
  const { backend, base } = cloudMetaProfilesTarget();
  return callCloudProxy<ActivateMetaProfileResponse>({
    backend,
    method: "POST",
    path: `${base}/${encodeURIComponent(name)}/activate`,
    body: {},
  });
}
