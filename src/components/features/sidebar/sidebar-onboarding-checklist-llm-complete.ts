import type { ProfileListResponse } from "#/api/profiles-service/profiles-service.api";
import type { Settings } from "#/types/settings";

function hasLlmProfileWithApiKey(
  profilesData: ProfileListResponse | undefined,
): boolean {
  return (
    profilesData?.profiles.some((profile) => profile.api_key_set === true) ??
    false
  );
}

function hasConfiguredLlmInSettings(settings: Settings | undefined): boolean {
  const llm = settings?.agent_settings?.llm as
    | { model?: unknown; auth_type?: unknown }
    | undefined;
  const hasModel = typeof llm?.model === "string" && llm.model.length > 0;
  const hasAuth =
    settings?.llm_api_key_set === true ||
    settings?.llm_api_key_is_set === true ||
    llm?.auth_type === "subscription";

  return hasModel && hasAuth;
}

export function isConfigureLlmChecklistItemComplete(
  settings: Settings | undefined,
  isLlmConfigured: boolean,
  isLlmConfiguredLoading: boolean,
  profilesData: ProfileListResponse | undefined,
  isProfilesLoading: boolean,
): boolean {
  if (hasLlmProfileWithApiKey(profilesData)) {
    return true;
  }

  if (isLlmConfigured) {
    return true;
  }

  if (hasConfiguredLlmInSettings(settings)) {
    return true;
  }

  if (isLlmConfiguredLoading || (isProfilesLoading && !profilesData)) {
    return false;
  }

  return false;
}
