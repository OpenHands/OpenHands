import { useQuery } from "@tanstack/react-query";
import { openHands } from "#/api/open-hands-axios";
import { useConfig } from "./use-config";

export const LLM_API_KEY_QUERY_KEY = "llm-api-key";

export interface LlmApiKeyResponse {
  key: string | null;
}

export function useLlmApiKey() {
  const { data: config } = useConfig();

  return useQuery({
    queryKey: [LLM_API_KEY_QUERY_KEY],
    enabled: config?.APP_MODE === "saas",
    queryFn: async () => {
      const { data } =
        await openHands.get<LlmApiKeyResponse>("/api/keys/llm/byor");
      return data;
    },
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });
}
