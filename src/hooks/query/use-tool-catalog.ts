import { useQuery } from "@tanstack/react-query";

import ToolCatalogService from "#/api/tool-catalog-service/tool-catalog-service.api";
import { agentProfileMayServeToolCatalog } from "#/api/agent-profiles-service/profile-field-support";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useAgentServerInfo } from "#/hooks/query/use-agent-server-info";
import {
  AGENT_PROFILES_RETRY_OPTIONS,
  CONFIG_CACHE_OPTIONS,
  TOOL_CATALOG_QUERY_KEYS,
} from "#/hooks/query/query-keys";

interface UseToolCatalogOptions {
  enabled?: boolean;
}

/** Tools the active backend offers; `null` data when it serves no catalog. */
export function useToolCatalog(options: UseToolCatalogOptions = {}) {
  const { backend } = useActiveBackend();
  const { data: serverInfo } = useAgentServerInfo();
  const mayServe = agentProfileMayServeToolCatalog(backend.kind, serverInfo);

  const query = useQuery({
    queryKey: [...TOOL_CATALOG_QUERY_KEYS.all, backend.id, backend.host],
    queryFn: ToolCatalogService.getCatalog,
    enabled: (options.enabled ?? true) && mayServe,
    ...CONFIG_CACHE_OPTIONS,
    ...AGENT_PROFILES_RETRY_OPTIONS,
    meta: { disableToast: true },
  });
  return {
    data: query.data,
    isError: query.isError,
    refetch: query.refetch,
    /** Whether the backend takes a `tools` selection from the picker. */
    supported: mayServe && query.data !== null,
  };
}
