import { useMutation, useQueryClient } from "@tanstack/react-query";
import ProviderConnectionsService from "#/api/provider-connections-service/provider-connections-service.api";
import {
  LLM_PROFILES_QUERY_KEYS,
  PROVIDER_CONNECTIONS_QUERY_KEYS,
} from "#/hooks/query/query-keys";

export function useDeleteProviderConnection() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => ProviderConnectionsService.delete(id),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: PROVIDER_CONNECTIONS_QUERY_KEYS.all,
      });
      await queryClient.invalidateQueries({
        queryKey: LLM_PROFILES_QUERY_KEYS.all,
      });
    },
    // Consumers handle errors with try-catch and manual toasts (e.g. the 409
    // returned when a profile still references the connection).
    meta: { disableToast: true },
  });
}
