import { useMutation, useQueryClient } from "@tanstack/react-query";
import MetaProfilesService from "#/api/meta-profiles-service/meta-profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  META_PROFILES_QUERY_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";

export function useDeleteMetaProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (name: string) => MetaProfilesService.deleteMetaProfile(name),
    onSuccess: async () => {
      // Deleting the *active* meta-profile clears ``active_meta_profile`` in
      // settings (and detaches the classify_and_switch_llm tool from new
      // conversations), so refresh the settings caches too — mirroring
      // activation. Without this, settings can stay stale until reload.
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
