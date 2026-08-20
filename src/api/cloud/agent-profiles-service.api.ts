import type {
  AgentProfileSaveInput,
  AgentProfileListResponse,
  AgentProfileDetailResponse,
  AgentProfileMutationResponse,
  ActivateAgentProfileResponse,
} from "@openhands/typescript-client";
import { getActiveBackend } from "../backend-registry/active-store";
import type { Backend } from "../backend-registry/types";
import { callCloudProxy } from "./proxy";

/**
 * Cloud transport for the Agent Profiles surface. The cloud enterprise
 * app-server exposes the SAME `/api/agent-profiles` contract as the local
 * agent-server (paths + response shapes mirror `agent_profiles_router.py`),
 * but cloud calls authenticate with the backend's bearer token + `X-Org-Id`
 * and must go through {@link callCloudProxy} — not the direct
 * `AgentProfilesClient` (which speaks `X-Session-API-Key` to a local host).
 *
 * The org is resolved server-side from the auth session (`EFFECTIVE_ORG_ID`),
 * so — unlike org LLM profiles — no `{org_id}` path segment is needed here.
 *
 * Mirrors the pattern in `cloud/settings-service.api.ts`. Consumed by
 * `AgentProfilesService` on the cloud path.
 */

const BASE = "/api/agent-profiles";

function activeCloudBackend(): Backend {
  const active = getActiveBackend().backend;
  if (active.kind !== "cloud") {
    throw new Error("Cloud agent-profile call requires a cloud backend.");
  }
  return active;
}

export async function listCloudAgentProfiles(): Promise<AgentProfileListResponse> {
  return callCloudProxy<AgentProfileListResponse>({
    backend: activeCloudBackend(),
    method: "GET",
    path: BASE,
  });
}

export async function getCloudAgentProfile(
  name: string,
): Promise<AgentProfileDetailResponse> {
  return callCloudProxy<AgentProfileDetailResponse>({
    backend: activeCloudBackend(),
    method: "GET",
    path: `${BASE}/${encodeURIComponent(name)}`,
  });
}

export async function saveCloudAgentProfile(
  name: string,
  profile: AgentProfileSaveInput,
): Promise<AgentProfileMutationResponse> {
  return callCloudProxy<AgentProfileMutationResponse>({
    backend: activeCloudBackend(),
    method: "POST",
    path: `${BASE}/${encodeURIComponent(name)}`,
    body: profile,
  });
}

export async function deleteCloudAgentProfile(
  name: string,
): Promise<AgentProfileMutationResponse> {
  return callCloudProxy<AgentProfileMutationResponse>({
    backend: activeCloudBackend(),
    method: "DELETE",
    path: `${BASE}/${encodeURIComponent(name)}`,
  });
}

export async function renameCloudAgentProfile(
  name: string,
  newName: string,
): Promise<AgentProfileMutationResponse> {
  return callCloudProxy<AgentProfileMutationResponse>({
    backend: activeCloudBackend(),
    method: "POST",
    path: `${BASE}/${encodeURIComponent(name)}/rename`,
    body: { new_name: newName },
  });
}

export async function activateCloudAgentProfile(
  profileId: string,
): Promise<ActivateAgentProfileResponse> {
  return callCloudProxy<ActivateAgentProfileResponse>({
    backend: activeCloudBackend(),
    method: "POST",
    path: `${BASE}/${encodeURIComponent(profileId)}/activate`,
    body: {},
  });
}
