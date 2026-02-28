import { useQuery } from "@tanstack/react-query";
import ApiKeysClient from "#/api/api-keys";
import { useConfig } from "./use-config";

export const API_KEYS_QUERY_KEY = "api-keys";

export function useApiKeys() {
  const { data: config } = useConfig();

  return useQuery({
    queryKey: [API_KEYS_QUERY_KEY],
    enabled: config?.app_mode === "saas",
    queryFn: async () => {
      const keys = await ApiKeysClient.getApiKeys();
      return Array.isArray(keys) ? keys : [];
    },
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });
}
