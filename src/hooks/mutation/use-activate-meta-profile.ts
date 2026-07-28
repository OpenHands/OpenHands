import { useMutation, useQueryClient } from "@tanstack/react-query";
import MetaProfilesService from "#/api/meta-profiles-service/meta-profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  META_PROFILES_QUERY_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";

export function useActivateMetaProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (name: string) => MetaProfilesService.activateMetaProfile(name),
    onSuccess: async () => {
      // Activating a meta-profile sets ``active_meta_profile`` in settings,
      // which controls whether the classify_and_switch_llm tool is attached to
      // new conversations — so refresh the settings caches too.
      SettingsService.invalidateCache();
      await queryClient.invalidateQueries({
        queryKey: META_PROFILES_QUERY_KEYS.all,
      });
      await queryClient.invalidateQueries({
        queryKey: SETTINGS_QUERY_KEYS.personal(),
      });
    },
    // Consumers handle errors with try-catch and manual toasts; disable global toast
    meta: { disableToast: true },
  });
}
