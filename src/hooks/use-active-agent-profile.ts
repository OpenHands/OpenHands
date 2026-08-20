import type { AgentKind } from "@openhands/typescript-client";
import type { AgentProfileSummary } from "#/api/agent-profiles-service/agent-profiles-service.api";
import { useAgentProfiles } from "#/hooks/query/use-agent-profiles";

/**
 * The agent profile the user has activated — the authoritative "current agent"
 * now that Settings → Agent IS the profile library (#1571). `activate` is
 * pointer-only and never writes `settings.agent_settings`, so the active
 * profile — not the global agent settings — is the source of truth for what
 * kind of agent (OpenHands vs ACP) is in effect.
 */
export function useActiveAgentProfile(): {
  activeProfile: AgentProfileSummary | null;
  isLoading: boolean;
} {
  const { data, isLoading } = useAgentProfiles();
  const activeId = data?.active_agent_profile_id ?? null;
  const activeProfile =
    data?.profiles?.find((p) => p.id != null && p.id === activeId) ?? null;
  return { activeProfile, isLoading };
}

/**
 * The effective agent kind from the active profile. `undefined` while the
 * profile list is loading or when no profile is active — callers should fall
 * back to `settings.agent_settings.agent_kind` in that window to avoid a flash
 * of the wrong (OpenHands-default) UI.
 */
export function useActiveAgentKind(): AgentKind | undefined {
  const { activeProfile } = useActiveAgentProfile();
  return activeProfile?.agent_kind;
}
