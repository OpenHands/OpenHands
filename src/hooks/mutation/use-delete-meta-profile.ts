import { useMutation, useQueryClient } from "@tanstack/react-query";
import MetaProfilesService from "#/api/meta-profiles-service/meta-profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  routerDetached,
  updateEveryAgentProfile,
} from "#/hooks/mutation/sync-agent-profiles";
import {
  AGENT_PROFILES_QUERY_KEYS,
  META_PROFILES_QUERY_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";

export function useDeleteMetaProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (name: string) => {
      // A launch refuses a missing router, so detach it first.
      await updateEveryAgentProfile(queryClient, routerDetached(name), {
        throwOnError: true,
      });
      return MetaProfilesService.deleteMetaProfile(name);
    },
    onSuccess: async () => {
      // Deleting the *active* meta-profile clears ``active_meta_profile`` in
      // settings, so refresh the settings caches too — mirroring activation.
      SettingsService.invalidateCache();
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: META_PROFILES_QUERY_KEYS.all,
        }),
        queryClient.invalidateQueries({
          queryKey: SETTINGS_QUERY_KEYS.personal(),
        }),
        queryClient.invalidateQueries({
          queryKey: AGENT_PROFILES_QUERY_KEYS.all,
        }),
      ]);
    },
    // Consumers handle errors with try-catch and manual toasts; disable global toast
    meta: { disableToast: true },
  });
}
