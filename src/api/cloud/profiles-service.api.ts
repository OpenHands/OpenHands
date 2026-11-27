import type {
  ActivateProfileResponse,
  ProfileDetailResponse,
  ProfileListResponse,
  ProfileMutationResponse,
  SaveProfileRequest,
} from "@openhands/typescript-client";
import { getActiveBackend } from "../backend-registry/active-store";
import type { Backend } from "../backend-registry/types";
import { callCloudProxy } from "./proxy";

/**
 * Cloud LLM-profile service.
 *
 * Profile CRUD is routed to the org-scoped endpoints
 * `/api/organizations/{orgId}/profiles`, which enforce `EDIT_ORG_SETTINGS`
 * server-side — so a member's mutation is rejected with 403 even on a direct
 * API call, not just hidden in the UI. When the active cloud backend has no org
 * bound (legacy API keys), we fall back to the ungated per-user settings route
 * `/api/v1/settings/profiles`, where there is no org role to enforce against.
 *
 * The two routes share shapes except `get` (org returns `llm`, settings returns
 * `config`) and `activate` (org returns `llm`, settings returns `model`); both
 * are normalized to the SDK profile types below. Neither route exposes secrets.
 */

const SETTINGS_PROFILES_PATH = "/api/v1/settings/profiles";

/**
 * Resolve the backend + base path for the active cloud backend's profiles:
 * the org-gated route when an org is bound, else the per-user settings route.
 */
function cloudProfilesTarget(): { backend: Backend; base: string } {
  const { backend, orgId } = getActiveBackend();
  if (backend.kind !== "cloud") {
    throw new Error("Cloud profiles call requires a cloud backend.");
  }
  return {
    backend,
    base: orgId
      ? `/api/organizations/${encodeURIComponent(orgId)}/profiles`
      : SETTINGS_PROFILES_PATH,
  };
}

export async function fetchCloudProfiles(): Promise<ProfileListResponse> {
  const { backend, base } = cloudProfilesTarget();
  return callCloudProxy<ProfileListResponse>({
    backend,
    method: "GET",
    path: base,
  });
}

export async function fetchCloudProfile(
  name: string,
): Promise<ProfileDetailResponse> {
  const { backend, base } = cloudProfilesTarget();
  // Org returns `{ name, llm }`; settings returns `{ name, config, api_key_set }`.
  // Normalize to the SDK detail shape. Neither route exposes the key, and the
  // GUI never reads the detail's `api_key_set` (only list-item `api_key_set`).
  const result = await callCloudProxy<{
    name: string;
    config?: Record<string, unknown>;
    llm?: Record<string, unknown>;
    api_key_set?: boolean;
  }>({
    backend,
    method: "GET",
    path: `${base}/${encodeURIComponent(name)}`,
  });
  return {
    name: result.name,
    config: result.config ?? result.llm ?? {},
    api_key_set: result.api_key_set ?? false,
  };
}

export async function saveCloudProfile(
  name: string,
  request: SaveProfileRequest,
): Promise<ProfileMutationResponse> {
  const { backend, base } = cloudProfilesTarget();
  return callCloudProxy<ProfileMutationResponse>({
    backend,
    method: "POST",
    path: `${base}/${encodeURIComponent(name)}`,
    body: request,
  });
}

export async function deleteCloudProfile(
  name: string,
): Promise<ProfileMutationResponse> {
  const { backend, base } = cloudProfilesTarget();
  return callCloudProxy<ProfileMutationResponse>({
    backend,
    method: "DELETE",
    path: `${base}/${encodeURIComponent(name)}`,
  });
}

export async function renameCloudProfile(
  name: string,
  newName: string,
): Promise<ProfileMutationResponse> {
  const { backend, base } = cloudProfilesTarget();
  return callCloudProxy<ProfileMutationResponse>({
    backend,
    method: "POST",
    path: `${base}/${encodeURIComponent(name)}/rename`,
    body: { new_name: newName },
  });
}

export async function activateCloudProfile(
  name: string,
): Promise<ActivateProfileResponse> {
  const { backend, base } = cloudProfilesTarget();
  // Org returns `{ name, message, llm }`; settings returns `{ name, message,
  // model }`. The SDK type carries `llm_applied`; derive it from whichever the
  // route provided (consumers only read name/message — the hook just
  // invalidates caches).
  const result = await callCloudProxy<{
    name: string;
    message: string;
    model?: string | null;
    llm?: Record<string, unknown> | null;
  }>({
    backend,
    method: "POST",
    path: `${base}/${encodeURIComponent(name)}/activate`,
    body: {},
  });
  return {
    name: result.name,
    message: result.message,
    llm_applied: result.model != null || result.llm != null,
  };
}
