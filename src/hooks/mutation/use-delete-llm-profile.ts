import { useMutation, useQueryClient } from "@tanstack/react-query";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  LLM_PROFILES_QUERY_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import {
  llmProfileMovedOff,
  updateDefaultAgentProfile,
} from "#/hooks/mutation/sync-agent-profiles";

/**
 * `default` follows the active LLM profile, and the server refuses to delete
 * one an agent profile still uses: move `default` to the profile that will
 * become active first. The only profile left stays, since `default` needs one.
 */
async function moveDefaultOffLlmProfile(
  queryClient: ReturnType<typeof useQueryClient>,
  name: string,
): Promise<void> {
  if (getActiveBackend().backend.kind !== "local") return;
  let profiles: Awaited<
    ReturnType<typeof ProfilesService.listProfiles>
  >["profiles"] = [];
  try {
    profiles = (await ProfilesService.listProfiles())?.profiles ?? [];
  } catch {
    // Without the list the delete still runs; the server explains a refusal.
  }
  const remaining = profiles.filter((profile) => profile.name !== name);
  const next = remaining.find((profile) => profile.api_key_set) ?? remaining[0];
  if (!next) return;
  await updateDefaultAgentProfile(
    queryClient,
    llmProfileMovedOff(name, next.name),
    { throwOnError: true },
  );
}

export function useDeleteLlmProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (name: string) => {
      await moveDefaultOffLlmProfile(queryClient, name);
      return ProfilesService.deleteProfile(name);
    },
    onSuccess: async (_response, name) => {
      try {
        if (getActiveBackend().backend.kind === "local") {
          const settings = await SettingsService.getSettings();
          if (settings?.title_llm_profile === name) {
            await SettingsService.saveSettings({ title_llm_profile: null });
          }
        }
      } finally {
        // Invalidate the SettingsService internal cache so getSettingsForConversation
        // fetches fresh settings after the profile is deleted (backend may have
        // deactivated it or reset agent_settings.llm)
        SettingsService.invalidateCache();
        await queryClient.invalidateQueries({
          queryKey: LLM_PROFILES_QUERY_KEYS.all,
        });
        // Use personal() scope for consistency with other settings hooks
        await queryClient.invalidateQueries({
          queryKey: SETTINGS_QUERY_KEYS.personal(),
        });
      }
    },
    // Consumers handle errors with try-catch and manual toasts; disable global toast
    meta: { disableToast: true },
  });
}
