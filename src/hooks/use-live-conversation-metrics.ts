import { useMemo } from "react";
import useMetricsStore, { type MetricsState } from "#/stores/metrics-store";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConversationMetrics } from "#/hooks/query/use-conversation-metrics";

/**
 * Cost/token metrics and budget for the active conversation.
 * Prefers matching live WebSocket metrics, falling back to the REST snapshot
 * when no stats have arrived or the latest stats belong to a planning child.
 * Pass `enabled: false` to pause REST polling while the consumer is hidden.
 */
export function useLiveConversationMetrics(
  enabled: boolean = true,
): Pick<MetricsState, "cost" | "usage"> & { maxBudget: number | null } {
  // Per-field selectors rather than a bare `useMetricsStore()`: subscribing to
  // the whole store re-runs this hook on any store write, including the action
  // identity, and matches how `useContextWindowUsage` reads the same store.
  const sourceConversationId = useMetricsStore(
    (state) => state.sourceConversationId,
  );
  const storeCost = useMetricsStore((state) => state.cost);
  const storeUsage = useMetricsStore((state) => state.usage);
  const { data: conversation } = useActiveConversation();

  const { data: conversationMetrics, maxBudget } = useConversationMetrics(
    conversation?.id,
    conversation?.conversation_url,
    conversation?.session_api_key,
    enabled,
  );

  return useMemo(() => {
    if (
      sourceConversationId === conversation?.id &&
      (storeCost !== null || storeUsage !== null)
    ) {
      return {
        cost: storeCost,
        maxBudget,
        usage: storeUsage,
      };
    }

    if (conversationMetrics) {
      return {
        cost: conversationMetrics.accumulated_cost,
        maxBudget,
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
      cost: null,
      maxBudget,
      usage: null,
    };
  }, [
    conversation?.id,
    conversationMetrics,
    maxBudget,
    sourceConversationId,
    storeCost,
    storeUsage,
  ]);
}
