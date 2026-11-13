import { useQuery } from "@tanstack/react-query";

import SkillsService from "#/api/skills-service";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  AGENT_PROFILES_RETRY_OPTIONS,
  CONFIG_CACHE_OPTIONS,
  SKILL_CATALOG_QUERY_KEYS,
} from "#/hooks/query/query-keys";

interface UseSkillCatalogOptions {
  enabled?: boolean;
}

/** Skills a local backend's agent profiles can select. */
export function useSkillCatalog(options: UseSkillCatalogOptions = {}) {
  const { backend } = useActiveBackend();

  return useQuery({
    queryKey: [...SKILL_CATALOG_QUERY_KEYS.all, backend.id, backend.host],
    queryFn: SkillsService.getSkillCatalog,
    enabled: (options.enabled ?? true) && backend.kind !== "cloud",
    ...CONFIG_CACHE_OPTIONS,
    ...AGENT_PROFILES_RETRY_OPTIONS,
    meta: { disableToast: true },
  });
}
