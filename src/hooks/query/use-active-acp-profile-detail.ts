import { useQuery } from "@tanstack/react-query";
import type { ACPAgentProfile } from "@openhands/typescript-client";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useActiveAgentProfile } from "#/hooks/use-active-agent-profile";
import {
  CONFIG_CACHE_OPTIONS,
  AGENT_PROFILES_QUERY_KEYS,
  AGENT_PROFILES_RETRY_OPTIONS,
} from "./query-keys";

/**
 * Cache key for one profile's full detail. Lives under the agent-profiles
 * prefix on purpose: activate/save/delete invalidate that prefix, so the
 * detail refetches for free. (`useActivateAgentProfile.onMutate` also
 * `setQueriesData`s the prefix, writing a stray `active_agent_profile_id`
 * onto this entry — benign, consumers only read `.profile`.) Shared with
 * `useSwitchAcpModel`, which `ensureQueryData`s the same entry.
 */
export function agentProfileDetailQueryKey(
  backendId: string,
  orgId: string | null | undefined,
  name: string,
) {
  return [
    ...AGENT_PROFILES_QUERY_KEYS.all,
    backendId,
    orgId,
    "detail",
    name,
  ] as const;
}

/**
 * Full detail of the active AgentProfile when it is ACP and no conversation is
 * open. `AgentProfileSummary` carries no `acp_server`/`acp_model`, and profile
 * activation never writes `settings.agent_settings` — so the home-page ACP
 * model picker must read the active profile's own fields (the conversation
 * launch source) rather than the possibly-stale global agent settings.
 *
 * Returns the ACP-narrowed profile, or `null` while loading / when the active
 * profile isn't ACP / on legacy backends without the profiles surface (callers
 * fall back to `settings.agent_settings` there).
 */
export function useActiveAcpProfileDetail(): ACPAgentProfile | null {
  const { backend, orgId } = useActiveBackend();
  const { conversationId } = useOptionalConversationId();
  const { activeProfile } = useActiveAgentProfile();
  const activeAcpProfileName =
    activeProfile?.agent_kind === "acp" ? activeProfile.name : null;

  const { data } = useQuery({
    queryKey: agentProfileDetailQueryKey(
      backend.id,
      orgId,
      activeAcpProfileName ?? "",
    ),
    queryFn: () => AgentProfilesService.getProfile(activeAcpProfileName!),
    enabled: !conversationId && activeAcpProfileName != null,
    ...CONFIG_CACHE_OPTIONS,
    ...AGENT_PROFILES_RETRY_OPTIONS,
    meta: { disableToast: true },
  });

  const profile = data?.profile;
  return profile?.agent_kind === "acp" ? profile : null;
}
