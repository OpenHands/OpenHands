import { useQuery } from "@tanstack/react-query";

import ToolCatalogService from "#/api/tool-catalog-service/tool-catalog-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  AGENT_PROFILES_RETRY_OPTIONS,
  CONFIG_CACHE_OPTIONS,
  TOOL_CATALOG_QUERY_KEYS,
} from "#/hooks/query/query-keys";

interface UseToolCatalogOptions {
  enabled?: boolean;
}

/** Tools the active backend offers. */
export function useToolCatalog(options: UseToolCatalogOptions = {}) {
  const { backend } = useActiveBackend();

  return useQuery({
    queryKey: [...TOOL_CATALOG_QUERY_KEYS.all, backend.id, backend.host],
    queryFn: ToolCatalogService.getCatalog,
    enabled: options.enabled ?? true,
    ...CONFIG_CACHE_OPTIONS,
    ...AGENT_PROFILES_RETRY_OPTIONS,
    meta: { disableToast: true },
  });
}
