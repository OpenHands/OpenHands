import { useQuery } from "@tanstack/react-query";
import { AxiosError } from "axios";
import { openHands } from "#/api/open-hands-axios";
import { useConfig } from "./use-config";

export const LLM_API_KEY_QUERY_KEY = "llm-api-key";

export interface LlmApiKeyResponse {
  key: string | null;
}

export interface LlmApiKeyError {
  isPaymentRequired: boolean;
  message?: string;
}

export function useLlmApiKey() {
  const { data: config } = useConfig();

  const query = useQuery({
    queryKey: [LLM_API_KEY_QUERY_KEY],
    enabled: config?.app_mode === "saas",
    queryFn: async () => {
      const { data } =
        await openHands.get<LlmApiKeyResponse>("/api/keys/llm/byor");
      return data;
    },
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
    retry: (failureCount, error) => {
      // Don't retry on 402 Payment Required
      if (error instanceof AxiosError && error.response?.status === 402) {
        return false;
      }
      return failureCount < 3;
    },
    // Disable global error toast - we handle 402 errors in the UI
    meta: { disableToast: true },
  });

  // Check if the error is a 402 Payment Required
  const isPaymentRequired =
    query.error instanceof AxiosError && query.error.response?.status === 402;

  return {
    data: query.data,
    error: query.error,
    isLoading: query.isLoading,
    isPaymentRequired,
  };
}
