import { useMutation, useQueryClient } from "@tanstack/react-query";
import ProviderConnectionsService, {
  type UpdateProviderConnectionRequest,
} from "#/api/provider-connections-service/provider-connections-service.api";
import {
  LLM_PROFILES_QUERY_KEYS,
  PROVIDER_CONNECTIONS_QUERY_KEYS,
} from "#/hooks/query/query-keys";

interface UpdateProviderConnectionVariables {
  id: string;
  request: UpdateProviderConnectionRequest;
}

export function useUpdateProviderConnection() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, request }: UpdateProviderConnectionVariables) =>
      ProviderConnectionsService.update(id, request),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: PROVIDER_CONNECTIONS_QUERY_KEYS.all,
      });
      // Linked profiles report the connection's key presence via `api_key_set`,
      // so refresh the profile list too after a rotation or rename.
      await queryClient.invalidateQueries({
        queryKey: LLM_PROFILES_QUERY_KEYS.all,
      });
    },
    // Consumers handle errors with try-catch and manual toasts.
    meta: { disableToast: true },
  });
}
