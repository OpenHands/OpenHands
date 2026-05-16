import { useMutation, useQueryClient } from "@tanstack/react-query";
import { SecretsService } from "#/api/secrets-service";
import { Provider, ProviderToken } from "#/types/settings";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";

export const useAddGitProviders = () => {
  const queryClient = useQueryClient();
  const { organizationId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: ({
      providers,
    }: {
      providers: Record<Provider, ProviderToken>;
    }) => SecretsService.addGitProvider(providers),
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
