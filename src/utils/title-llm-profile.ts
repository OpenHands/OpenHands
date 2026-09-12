import type { ProfileListResponse } from "#/api/profiles-service/profiles-service.api";

/**
 * Resolve which LLM profile should generate the conversation title.
 *
 * Priority:
 *   1. `preference` — the user's explicit `title_llm_profile` setting.
 *   2. `agentLlmProfileRef` — the LLM pinned to the running agent profile,
 *      so title generation uses the same model as the agent when no explicit
 *      preference is set.
 *   3. `profiles.active_profile` — account-wide fallback.
 */
export function resolveTitleLlmProfile(
  preference: string | null | undefined,
  profiles: ProfileListResponse | undefined,
  agentLlmProfileRef?: string | null,
): string | undefined {
  if (!profiles) return undefined;

  const availableProfiles = new Set(
    profiles.profiles.map((profile) => profile.name),
  );
  if (preference && availableProfiles.has(preference)) {
    return preference;
  }
  if (agentLlmProfileRef && availableProfiles.has(agentLlmProfileRef)) {
    return agentLlmProfileRef;
  }
  if (
    profiles.active_profile &&
    availableProfiles.has(profiles.active_profile)
  ) {
    return profiles.active_profile;
  }
  return undefined;
}
