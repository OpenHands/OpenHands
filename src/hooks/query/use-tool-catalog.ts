import { useQuery } from "@tanstack/react-query";

import type { Backend } from "#/api/backend-registry/types";
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

/** One cache entry per backend: the catalog is that backend's own tool list. */
export function toolCatalogQueryOptions(backend: Backend) {
  return {
    queryKey: [...TOOL_CATALOG_QUERY_KEYS.all, backend.id, backend.host],
    queryFn: ToolCatalogService.getCatalog,
    ...CONFIG_CACHE_OPTIONS,
    ...AGENT_PROFILES_RETRY_OPTIONS,
    meta: { disableToast: true },
  };
}

/** Tools the active backend offers. */
export function useToolCatalog(options: UseToolCatalogOptions = {}) {
  const { backend } = useActiveBackend();

  return useQuery({
    ...toolCatalogQueryOptions(backend),
    enabled: options.enabled ?? true,
  });
}
