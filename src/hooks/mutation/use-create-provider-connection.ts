import { useMutation, useQueryClient } from "@tanstack/react-query";
import ProviderConnectionsService, {
  type CreateProviderConnectionRequest,
} from "#/api/provider-connections-service/provider-connections-service.api";
import { PROVIDER_CONNECTIONS_QUERY_KEYS } from "#/hooks/query/query-keys";

export function useCreateProviderConnection() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: CreateProviderConnectionRequest) =>
      ProviderConnectionsService.create(request),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: PROVIDER_CONNECTIONS_QUERY_KEYS.all,
      });
    },
    // Consumers handle errors with try-catch and manual toasts.
    meta: { disableToast: true },
  });
}
