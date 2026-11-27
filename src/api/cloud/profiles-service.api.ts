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
 * The combined options endpoint lists personal and inherited organization
 * profiles. CRUD targets only the personal `/api/v1/settings/profiles` routes;
 * organization profiles remain selectable but are not editable here. Older
 * cloud backends fall back to the personal list endpoint. Neither route exposes
 * secrets.
 */

const SETTINGS_PROFILES_PATH = "/api/v1/settings/profiles";

function activeCloudBackend(): Backend {
  const { backend } = getActiveBackend();
  if (backend.kind !== "cloud") {
    throw new Error("Cloud profiles call requires a cloud backend.");
  }
  return backend;
}

function personalProfilesTarget(): { backend: Backend; base: string } {
  return { backend: activeCloudBackend(), base: SETTINGS_PROFILES_PATH };
}

interface CloudAgentProfileOptions {
  llm_profiles?: Array<{
    name: string;
    scope?: "personal" | "organization";
    model?: string | null;
    base_url?: string | null;
    api_key_set?: boolean;
    provider_connection_id?: string | null;
    writable?: boolean;
    deletable?: boolean;
  }>;
}

export async function fetchCloudProfiles(): Promise<ProfileListResponse> {
  const backend = activeCloudBackend();
  try {
    const options = await callCloudProxy<CloudAgentProfileOptions>({
      backend,
      method: "GET",
      path: "/api/v1/agent-profile/options",
    });
    if (Array.isArray(options.llm_profiles)) {
      return {
        profiles: options.llm_profiles.map((profile) => ({
          name: profile.name,
          model: profile.model ?? null,
          base_url: profile.base_url ?? null,
          api_key_set: profile.api_key_set ?? false,
          provider_connection_id: profile.provider_connection_id ?? null,
          ...(profile.scope && { scope: profile.scope }),
          ...(profile.writable !== undefined && { writable: profile.writable }),
          ...(profile.deletable !== undefined && {
            deletable: profile.deletable,
          }),
        })),
        active_profile: null,
      };
    }
  } catch {
    // Older cloud backends do not expose the combined options endpoint yet.
  }
  return callCloudProxy<ProfileListResponse>({
    backend,
    method: "GET",
    path: SETTINGS_PROFILES_PATH,
  });
}

export async function fetchCloudProfile(
  name: string,
): Promise<ProfileDetailResponse> {
  const { backend, base } = personalProfilesTarget();
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
  const { backend, base } = personalProfilesTarget();
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
  const { backend, base } = personalProfilesTarget();
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
  const { backend, base } = personalProfilesTarget();
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
  const { backend, base } = personalProfilesTarget();
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
