import { useQuery } from "@tanstack/react-query";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { CONFIG_CACHE_OPTIONS, LLM_PROFILES_QUERY_KEYS } from "./query-keys";

export { LLM_PROFILES_QUERY_KEYS };

interface UseLlmProfilesOptions {
  enabled?: boolean;
}

export function useLlmProfiles(options: UseLlmProfilesOptions = {}) {
  const { backend, orgId } = useActiveBackend();

  return useQuery({
    // Include backend identity to prevent cache pollution when switching backends
    queryKey: [...LLM_PROFILES_QUERY_KEYS.all, backend.id, orgId],
    queryFn: ProfilesService.listProfiles,
    ...CONFIG_CACHE_OPTIONS,
    enabled: options.enabled ?? true,
    meta: { disableToast: true },
  });
}
