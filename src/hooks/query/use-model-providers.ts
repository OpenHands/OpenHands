import { useQuery } from "@tanstack/react-query";
import ModelProvidersService from "#/api/model-providers-service";
import { MODEL_PROVIDERS_QUERY_KEYS } from "./query-keys";

/**
 * Lists configured model providers (connect a provider once → manage its
 * models under it). Disabled on cloud in this release; the cloud mirror is a
 * follow-up.
 */
export function useModelProviders({
  enabled = true,
}: { enabled?: boolean } = {}) {
  return useQuery({
    queryKey: MODEL_PROVIDERS_QUERY_KEYS.all,
    queryFn: ModelProvidersService.listProviders,
    enabled,
    retry: false,
    refetchOnWindowFocus: false,
    staleTime: 1000 * 60 * 5,
    meta: { disableToast: true },
  });
}
