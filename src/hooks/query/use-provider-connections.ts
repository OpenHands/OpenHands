import { useQuery } from "@tanstack/react-query";
import ProviderConnectionsService from "#/api/provider-connections-service";
import { PROVIDER_CONNECTIONS_QUERY_KEYS } from "./query-keys";

/**
 * Lists Provider Connections (connect a vendor once → pick from its models).
 * Disabled on cloud in this release; the cloud mirror is a follow-up.
 */
export function useProviderConnections({
  enabled = true,
}: { enabled?: boolean } = {}) {
  return useQuery({
    queryKey: PROVIDER_CONNECTIONS_QUERY_KEYS.all,
    queryFn: ProviderConnectionsService.listConnections,
    enabled,
    retry: false,
    refetchOnWindowFocus: false,
    staleTime: 1000 * 60 * 5,
    meta: { disableToast: true },
  });
}
