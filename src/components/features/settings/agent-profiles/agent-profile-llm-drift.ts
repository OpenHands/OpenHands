import {
  type AgentProfile,
  type AgentProfileSummary,
} from "#/api/agent-profiles-service/agent-profiles-service.api";

/**
 * The LLM profile a home-launched conversation will really run on when it
 * differs from the agent profile's own `llm_profile_ref`, or `null` when the
 * two agree (the common case).
 *
 * Only the ACTIVE profile can drift. A home launch passes no explicit profile
 * id, so `useCreateConversation` resolves the active profile and — on local
 * backends — downgrades to an `agent_settings` launch whenever the
 * account-wide active LLM profile differs from the pinned ref (#16539), and
 * unconditionally for the seeded `default` profile (#16193). Either way the
 * launch runs the active LLM profile while the row still advertises the pinned
 * ref, and nothing surfaces the disagreement (#16265). An inactive profile's
 * ref stays authoritative for the in-conversation picker, so it never drifts.
 *
 * `activeLlmProfile` is the caller's signal that the downgrade applies: pass
 * `null` on cloud, where the pinned ref IS what launches (the server resolves
 * by profile id), and for a profile that fails {@link allowsAgentSettingsLaunch}.
 */
export function getAgentProfileLlmDrift(
  profile: AgentProfileSummary,
  isActive: boolean,
  activeLlmProfile: string | null,
): string | null {
  if (!isActive || !activeLlmProfile) return null;
  // ACP profiles own their LLM via the subprocess and carry no ref to drift.
  if (profile.agent_kind !== "openhands" || !profile.llm_profile_ref) {
    return null;
  }
  return profile.llm_profile_ref === activeLlmProfile ? null : activeLlmProfile;
}

/**
 * Whether `useCreateConversation` may downgrade this profile's home launch to
 * `agent_settings`. A profile with a secret scope (`secret_refs` is an array,
 * even an empty one) always keeps its profile id — an `agent_settings` launch
 * carries no profile identity for Agent Server to enforce the allow-list
 * against — so its pinned `llm_profile_ref` is what runs and it cannot drift.
 * An unread profile (`undefined`) fails closed, as the launch does.
 */
export function allowsAgentSettingsLaunch(
  profile: AgentProfile | undefined,
): boolean {
  return profile !== undefined && !Array.isArray(profile.secret_refs);
}
