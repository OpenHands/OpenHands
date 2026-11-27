import { useMutation, useQueryClient } from "@tanstack/react-query";
import ProfilesService, {
  type SaveProfileRequest,
} from "#/api/profiles-service/profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  LLM_PROFILES_QUERY_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";

interface SaveLlmProfileVariables {
  name: string;
  request: SaveProfileRequest;
}

export function useSaveLlmProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ name, request }: SaveLlmProfileVariables) =>
      ProfilesService.saveProfile(name, request),
    onSuccess: async () => {
      // Invalidate SettingsService internal cache to ensure fresh settings
      // for new conversations (especially if saving the active profile)
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
