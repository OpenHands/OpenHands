import { useQuery } from "@tanstack/react-query";
import OptionService from "#/api/option-service/option-service.api";
import { useIsOnIntermediatePage } from "#/hooks/use-is-on-intermediate-page";
import { QUERY_KEYS, CONFIG_CACHE_OPTIONS } from "./query-keys";

interface UseConfigOptions {
  enabled?: boolean;
}

export const useConfig = (options?: UseConfigOptions) => {
  const isOnIntermediatePage = useIsOnIntermediatePage();

  return useQuery({
    queryKey: QUERY_KEYS.WEB_CLIENT_CONFIG,
    queryFn: OptionService.getConfig,
    ...CONFIG_CACHE_OPTIONS,
    enabled: options?.enabled ?? !isOnIntermediatePage,
  });
};
