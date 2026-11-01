import { useMutation, useQueryClient } from "@tanstack/react-query";
import MetaProfilesService, {
  type MetaProfile,
} from "#/api/meta-profiles-service/meta-profiles-service.api";
import { META_PROFILES_QUERY_KEYS } from "#/hooks/query/query-keys";

interface SaveMetaProfileVariables {
  name: string;
  config: MetaProfile;
}

export function useSaveMetaProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ name, config }: SaveMetaProfileVariables) =>
      MetaProfilesService.saveMetaProfile(name, config),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: META_PROFILES_QUERY_KEYS.all,
      });
    },
    // Consumers handle errors with try-catch and manual toasts; disable global toast
    meta: { disableToast: true },
  });
}
