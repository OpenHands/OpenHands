import { useMutation, useQueryClient } from "@tanstack/react-query";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  LLM_PROFILES_QUERY_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";
import { getActiveBackend } from "#/api/backend-registry/active-store";

interface RenameLlmProfileVariables {
  name: string;
  newName: string;
}

export function useRenameLlmProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ name, newName }: RenameLlmProfileVariables) =>
      ProfilesService.renameProfile(name, newName),
    onSuccess: async (_response, { name, newName }) => {
      if (getActiveBackend().backend.kind === "local") {
        const settings = await SettingsService.getSettings();
        if (settings?.title_llm_profile === name) {
          await SettingsService.saveSettings({
            title_llm_profile: newName,
          });
        }
      }
      // Invalidate SettingsService internal cache to ensure fresh settings
      // (backend references in agent_settings may change when renaming active profile)
      SettingsService.invalidateCache();
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
