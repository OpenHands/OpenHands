import { useMutation, useQueryClient } from "@tanstack/react-query";
import LLMSubscriptionService from "#/api/llm-subscription-service";
import { LLM_SUBSCRIPTION_QUERY_KEYS } from "#/hooks/query/query-keys";

export function useStartOpenAISubscriptionLogin() {
  return useMutation({
    mutationFn: LLMSubscriptionService.startOpenAIDeviceLogin,
  });
}

export function usePollOpenAISubscriptionLogin() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: LLMSubscriptionService.pollOpenAIDeviceLogin,
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: LLM_SUBSCRIPTION_QUERY_KEYS.openaiStatus,
      });
    },
  });
}

export function useLogoutOpenAISubscription() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: LLMSubscriptionService.logoutOpenAI,
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: LLM_SUBSCRIPTION_QUERY_KEYS.openaiStatus,
      });
    },
  });
}
