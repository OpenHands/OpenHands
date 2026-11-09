import { useMutation, useQueryClient } from "@tanstack/react-query";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  LLM_PROFILES_QUERY_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";
import {
  llmProfileMirrored,
  updateDefaultAgentProfile,
} from "#/hooks/mutation/sync-agent-profiles";

export function useActivateLlmProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (name: string) => ProfilesService.activateProfile(name),
    onSuccess: async (_data, name) => {
      updateDefaultAgentProfile(queryClient, llmProfileMirrored(name));
      // Invalidate the SettingsService internal cache so getSettingsForConversation
      // fetches fresh settings with the newly activated profile's LLM config
      SettingsService.invalidateCache();
      // Invalidate profiles list to refresh active_profile
      await queryClient.invalidateQueries({
        queryKey: LLM_PROFILES_QUERY_KEYS.all,
      });
      // Use personal() scope for consistency with other settings hooks
      await queryClient.invalidateQueries({
        queryKey: SETTINGS_QUERY_KEYS.personal(),
      });
    },
    // Consumers handle errors with try-catch and manual toasts; disable global toast
    meta: { disableToast: true },
  });
}
