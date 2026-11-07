import { useQuery } from "@tanstack/react-query";
import ProviderConnectionsService from "#/api/provider-connections-service/provider-connections-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  CONFIG_CACHE_OPTIONS,
  PROVIDER_CONNECTIONS_QUERY_KEYS,
} from "./query-keys";

export { PROVIDER_CONNECTIONS_QUERY_KEYS };

/**
 * Provider connections are available on the local agent-server
 * (`/api/llm/provider-connections`) and on cloud when an org is bound
 * (`/api/organizations/{orgId}/provider-connections`). The query is disabled
 * only for a cloud backend without an org (legacy API keys), where the
 * org-scoped route cannot be addressed — there it returns no data so the
 * connections UI hides itself rather than firing an unaddressable request.
 */
export function useProviderConnections() {
  const { backend, orgId } = useActiveBackend();
  const isLocal = backend.kind === "local";
  const isCloudWithOrg = backend.kind === "cloud" && !!orgId;
  const enabled = isLocal || isCloudWithOrg;

  return useQuery({
    queryKey: [...PROVIDER_CONNECTIONS_QUERY_KEYS.all, backend.id, orgId],
    queryFn: ProviderConnectionsService.list,
    ...CONFIG_CACHE_OPTIONS,
    enabled,
    meta: { disableToast: true },
  });
}
