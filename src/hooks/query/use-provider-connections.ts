import { useQuery } from "@tanstack/react-query";
import ProviderConnectionsService from "#/api/provider-connections-service/provider-connections-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  CONFIG_CACHE_OPTIONS,
  PROVIDER_CONNECTIONS_QUERY_KEYS,
} from "./query-keys";

export { PROVIDER_CONNECTIONS_QUERY_KEYS };

/**
 * Provider connections live only on the local agent-server. On cloud backends
 * the query stays disabled and returns no data, so the connections UI hides
 * itself rather than firing a request that would 404.
 */
export function useProviderConnections() {
  const { backend } = useActiveBackend();
  const isLocal = backend.kind === "local";

  return useQuery({
    queryKey: [...PROVIDER_CONNECTIONS_QUERY_KEYS.all, backend.id],
    queryFn: ProviderConnectionsService.list,
    ...CONFIG_CACHE_OPTIONS,
    enabled: isLocal,
    meta: { disableToast: true },
  });
}
