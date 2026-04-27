import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { SecretsService } from "#/api/secrets-service";

export const useDeleteGitProviders = () => {
  const queryClient = useQueryClient();
  const { organizationId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: () => SecretsService.deleteGitProviders(),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["settings", "personal", organizationId],
      });
    },
    meta: {
      disableToast: true,
    },
  });
};
