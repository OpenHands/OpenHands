import { useQuery } from "@tanstack/react-query";
import MetaProfilesService from "#/api/meta-profiles-service/meta-profiles-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { CONFIG_CACHE_OPTIONS, META_PROFILES_QUERY_KEYS } from "./query-keys";

export { META_PROFILES_QUERY_KEYS };

interface UseMetaProfilesOptions {
  enabled?: boolean;
}

export function useMetaProfiles(options: UseMetaProfilesOptions = {}) {
  const { backend, orgId } = useActiveBackend();

  return useQuery({
    // Include backend identity to prevent cache pollution when switching backends
    queryKey: [...META_PROFILES_QUERY_KEYS.all, backend.id, orgId],
    queryFn: MetaProfilesService.listMetaProfiles,
    ...CONFIG_CACHE_OPTIONS,
    enabled: options.enabled ?? true,
    meta: { disableToast: true },
  });
}
