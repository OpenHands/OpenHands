import { useQuery } from "@tanstack/react-query";

import ToolCatalogService, {
  type DraftAgentProfile,
} from "#/api/tool-catalog-service/tool-catalog-service.api";
import { agentProfileSupportsToolCatalog } from "#/api/agent-profiles-service/profile-field-support";
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
    enabled: (options.enabled ?? true) && agentProfileSupportsToolCatalog(),
    ...CONFIG_CACHE_OPTIONS,
    ...AGENT_PROFILES_RETRY_OPTIONS,
    meta: { disableToast: true },
  });
}

interface UseResolvedProfileToolsOptions {
  draft: DraftAgentProfile | null;
  enabled?: boolean;
}

/**
 * Tool names a draft profile would launch with, as the server resolves them.
 *
 * Keyed by the draft itself: the standard set depends on server-side defaulting
 * and on what the runtime can run, so it is re-read whenever those inputs move
 * rather than cached against the profile name alone.
 */
export function useResolvedProfileTools({
  draft,
  enabled = true,
}: UseResolvedProfileToolsOptions) {
  const { backend } = useActiveBackend();

  return useQuery({
    queryKey: TOOL_CATALOG_QUERY_KEYS.resolved(backend.id, draft),
    queryFn: () => ToolCatalogService.getResolvedToolNames(draft!),
    enabled: enabled && draft !== null && agentProfileSupportsToolCatalog(),
    ...AGENT_PROFILES_RETRY_OPTIONS,
    meta: { disableToast: true },
  });
}
