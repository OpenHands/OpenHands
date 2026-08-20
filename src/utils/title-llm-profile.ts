import type { ProfileListResponse } from "#/api/profiles-service/profiles-service.api";

export function resolveTitleLlmProfile(
  preference: string | null | undefined,
  profiles: ProfileListResponse | undefined,
): string | undefined {
  if (!profiles) return undefined;

  const availableProfiles = new Set(
    profiles.profiles.map((profile) => profile.name),
  );
  if (preference && availableProfiles.has(preference)) {
    return preference;
  }
  if (
    profiles.active_profile &&
    availableProfiles.has(profiles.active_profile)
  ) {
    return profiles.active_profile;
  }
  return undefined;
}
