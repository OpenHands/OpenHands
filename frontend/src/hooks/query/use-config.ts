import { useQuery } from "@tanstack/react-query";
import OptionService from "#/api/option-service/option-service.api";
import { useIsOnIntermediatePage } from "#/hooks/use-is-on-intermediate-page";

export const WEB_CLIENT_CONFIG_QUERY_KEY = ["web-client-config"] as const;

interface UseConfigOptions {
  enabled?: boolean;
}

export const useConfig = (options?: UseConfigOptions) => {
  const isOnIntermediatePage = useIsOnIntermediatePage();

  return useQuery({
    queryKey: WEB_CLIENT_CONFIG_QUERY_KEY,
    queryFn: OptionService.getConfig,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes,
    enabled: options?.enabled ?? !isOnIntermediatePage,
  });
};
