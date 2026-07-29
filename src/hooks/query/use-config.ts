import { useQuery } from "@tanstack/react-query";
import {
  isAgentServerUnavailableError,
  isAgentServerAuthError,
} from "#/api/agent-server-compatibility";
import OptionService from "#/api/option-service/option-service.api";
import { QUERY_KEYS, CONFIG_CACHE_OPTIONS } from "./query-keys";

interface UseConfigOptions {
  enabled?: boolean;
}

export const useConfig = (options?: UseConfigOptions) =>
  useQuery({
    queryKey: QUERY_KEYS.WEB_CLIENT_CONFIG,
    queryFn: OptionService.getConfig,
    retry: (failureCount, error) =>
      !isAgentServerUnavailableError(error) &&
      !isAgentServerAuthError(error) &&
      failureCount < 3,
    meta: { disableToast: true },
    ...CONFIG_CACHE_OPTIONS,
    enabled: options?.enabled,
  });
