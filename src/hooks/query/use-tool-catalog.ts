import { useQuery } from "@tanstack/react-query";

import ToolCatalogService from "#/api/tool-catalog-service/tool-catalog-service.api";
import { agentProfileMayServeToolCatalog } from "#/api/agent-profiles-service/profile-field-support";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  AGENT_PROFILES_RETRY_OPTIONS,
  CONFIG_CACHE_OPTIONS,
  TOOL_CATALOG_QUERY_KEYS,
} from "#/hooks/query/query-keys";

interface UseToolCatalogOptions {
  enabled?: boolean;
}

/** Tools the active backend offers for configuring an agent. */
export function useToolCatalog(options: UseToolCatalogOptions = {}) {
  const { backend } = useActiveBackend();

  return useQuery({
    queryKey: [...TOOL_CATALOG_QUERY_KEYS.all, backend.id],
    queryFn: ToolCatalogService.getCatalog,
    enabled: (options.enabled ?? true) && agentProfileMayServeToolCatalog(),
    ...CONFIG_CACHE_OPTIONS,
    ...AGENT_PROFILES_RETRY_OPTIONS,
    meta: { disableToast: true },
  });
}
