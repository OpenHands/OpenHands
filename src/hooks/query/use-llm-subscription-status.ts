import { useQuery } from "@tanstack/react-query";
import LLMSubscriptionService from "#/api/llm-subscription-service";
import { LLM_SUBSCRIPTION_QUERY_KEYS } from "#/hooks/query/query-keys";

export function useOpenAISubscriptionStatus({
  enabled = true,
}: { enabled?: boolean } = {}) {
  return useQuery({
    queryKey: LLM_SUBSCRIPTION_QUERY_KEYS.openaiStatus,
    queryFn: LLMSubscriptionService.getOpenAIStatus,
    enabled,
    retry: false,
    refetchOnWindowFocus: false,
    staleTime: 1000 * 60 * 5,
    meta: {
      disableToast: true,
    },
  });
}
