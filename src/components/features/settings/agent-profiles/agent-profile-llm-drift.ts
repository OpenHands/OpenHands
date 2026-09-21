import { type AgentProfileSummary } from "#/api/agent-profiles-service/agent-profiles-service.api";

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
 * `activeLlmProfile` is the caller's local-only signal: pass `null` on cloud,
 * where the pinned ref IS what launches (the server resolves by profile id).
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
