import { useQuery } from "@tanstack/react-query";
import ApiKeysClient from "#/api/api-keys";
import { useConfig } from "./use-config";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const API_KEYS_QUERY_KEY = "api-keys";

export function useApiKeys() {
  const { data: config } = useConfig();
  const { organizationId } = useSelectedOrganizationId();

  return useQuery({
    queryKey: [API_KEYS_QUERY_KEY, organizationId],
    enabled: config?.app_mode === "saas" && !!organizationId,
    queryFn: async () => {
      const keys = await ApiKeysClient.getApiKeys();
      return Array.isArray(keys) ? keys : [];
    },
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });
}
