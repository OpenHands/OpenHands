import type { ProfileListResponse } from "#/api/profiles-service/profiles-service.api";

export function resolveTitleLlmProfile(
  preference: string | null | undefined,
  profiles: ProfileListResponse | undefined,
  agentProfileLlmRef?: string | null,
): string | undefined {
  if (!profiles) return undefined;

  const availableProfiles = new Set(
    profiles.profiles.map((profile) => profile.name),
  );
  if (preference && availableProfiles.has(preference)) {
    return preference;
  }
  // Without an explicit preference the title should come from the same LLM the
  // agent runs on, which is the profile's pinned ref rather than the
  // account-wide active one.
  if (agentProfileLlmRef && availableProfiles.has(agentProfileLlmRef)) {
    return agentProfileLlmRef;
  }
  if (
    profiles.active_profile &&
    availableProfiles.has(profiles.active_profile)
  ) {
    return profiles.active_profile;
  }
  return undefined;
}
