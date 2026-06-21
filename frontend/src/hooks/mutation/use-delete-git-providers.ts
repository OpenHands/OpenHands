import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { SecretsService } from "#/api/secrets-service";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";

export const useDeleteGitProviders = () => {
  const queryClient = useQueryClient();
  const { organizationId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: () => SecretsService.deleteGitProviders(),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: SETTINGS_QUERY_KEYS.personal(organizationId),
      });
    },
    meta: {
      disableToast: true,
    },
  });
};
