import { useMutation, useQueryClient } from "@tanstack/react-query";
import { SecretsService } from "#/api/secrets-service";
import { Provider, ProviderToken } from "#/types/settings";
import { useTracking } from "#/hooks/use-tracking";

export const useAddGitProviders = () => {
  const queryClient = useQueryClient();
  const { trackGitProviderConnected } = useTracking();

  return useMutation({
    mutationFn: ({
      providers,
    }: {
      providers: Record<Provider, ProviderToken>;
    }) => SecretsService.addGitProvider(providers),
    onSuccess: async (_, { providers }) => {
      // Track which providers were connected (filter out empty tokens)
      const connectedProviders = Object.entries(providers)
        .filter(([, value]) => value.token && value.token.trim() !== "")
        .map(([key]) => key);

      if (connectedProviders.length > 0) {
        trackGitProviderConnected({
          providers: connectedProviders,
        });
      }

      await queryClient.invalidateQueries({ queryKey: ["settings"] });
    },
    meta: {
      disableToast: true,
    },
  });
};
