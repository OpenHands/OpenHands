import { useQuery } from "@tanstack/react-query";
import OptionService from "#/api/option-service/option-service.api";
import { useIsOnIntermediatePage } from "#/hooks/use-is-on-intermediate-page";
import { queryClient } from "#/query-client-config";
import { WebClientConfig } from "#/api/option-service/option.types";

// Centralized query key for web client config
export const WEB_CLIENT_CONFIG_KEY = ["web-client-config"];

interface UseConfigOptions {
  enabled?: boolean;
}

/**
 * Hook for accessing web client config in React components.
 * Uses TanStack Query with 5-minute stale time and 15-minute garbage collection.
 */
export const useConfig = (options?: UseConfigOptions) => {
  const isOnIntermediatePage = useIsOnIntermediatePage();

  return useQuery({
    queryKey: WEB_CLIENT_CONFIG_KEY,
    queryFn: OptionService.getConfig,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
    enabled: options?.enabled ?? !isOnIntermediatePage,
  });
};

/**
 * Fetch web client config for use in clientLoaders/serverLoaders.
 * Uses TanStack Query's fetchQuery which properly handles caching.
 */
export const fetchConfig = (): Promise<WebClientConfig> =>
  queryClient.fetchQuery({
    queryKey: WEB_CLIENT_CONFIG_KEY,
    queryFn: OptionService.getConfig,
  });
