import { useMutation, useQueryClient } from "@tanstack/react-query";
import { SecretsService } from "#/api/secrets-service";

export const useDeleteProviderToken = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (provider: string) =>
      SecretsService.deleteProviderToken(provider),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["settings"] });
      queryClient.invalidateQueries({ queryKey: ["user"] });
    },
  });
};
