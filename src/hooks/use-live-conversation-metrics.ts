import { useMemo } from "react";
import useMetricsStore, { type MetricsState } from "#/stores/metrics-store";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConversationMetrics } from "#/hooks/query/use-conversation-metrics";

/**
 * Combined cost/token metrics for the active conversation.
 *
 * Prefers the REST snapshot (`GET /api/conversations/{id}` stats, combined
 * across all usage ids and refreshed on a 30s interval by
 * {@link useConversationMetrics}) and falls back to the live WebSocket
 * metrics store, which is updated in real time as agent events stream in.
 *
 * Pass `enabled: false` to pause the REST polling while the consumer is
 * hidden (e.g. a closed modal).
 */
export function useLiveConversationMetrics(
  enabled: boolean = true,
): MetricsState {
  const storeMetrics = useMetricsStore();
  const { data: conversation } = useActiveConversation();

  const { data: conversationMetrics } = useConversationMetrics(
    conversation?.id,
    conversation?.conversation_url,
    conversation?.session_api_key,
    enabled,
  );

  return useMemo(() => {
    if (conversationMetrics) {
      return {
        cost: conversationMetrics.accumulated_cost,
        max_budget_per_task: conversationMetrics.max_budget_per_task,
        usage: conversationMetrics.accumulated_token_usage
          ? {
              prompt_tokens:
                conversationMetrics.accumulated_token_usage.prompt_tokens ?? 0,
              completion_tokens:
                conversationMetrics.accumulated_token_usage.completion_tokens ??
                0,
              cache_read_tokens:
                conversationMetrics.accumulated_token_usage.cache_read_tokens ??
                0,
              cache_write_tokens:
                conversationMetrics.accumulated_token_usage
                  .cache_write_tokens ?? 0,
              context_window:
                conversationMetrics.accumulated_token_usage.context_window ?? 0,
              per_turn_token:
                conversationMetrics.accumulated_token_usage.per_turn_token ?? 0,
            }
          : null,
      };
    }

    return {
      cost: storeMetrics.cost,
      max_budget_per_task: storeMetrics.max_budget_per_task,
      usage: storeMetrics.usage,
    };
  }, [conversationMetrics, storeMetrics]);
}
