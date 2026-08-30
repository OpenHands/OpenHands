import { useMemo } from "react";
import useMetricsStore, { type MetricsState } from "#/stores/metrics-store";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConversationMetrics } from "#/hooks/query/use-conversation-metrics";
import type { MetricsSnapshot } from "#/api/conversation-service/agent-server-conversation-service.types";

/**
 * Combined cost/token metrics for the active conversation.
 *
 * Prefers the live WebSocket metrics store, which updates in real time as
 * agent events stream in, and falls back to the REST snapshot
 * (`GET /api/conversations/{id}` stats, combined across all usage ids and
 * refreshed on a 30s interval by {@link useConversationMetrics}) before the
 * first stats arrive. The store is reset on conversation switch (see
 * `ConversationWebSocketProvider`), so its contents always belong to the
 * active conversation — preferring it cannot leak a previous
 * conversation's figures, and preferring REST would lag the display up to
 * a poll interval behind what the user just watched happen.
 *
 * Pass `enabled: false` to pause the REST polling while the consumer is
 * hidden (e.g. a closed modal).
 */
export function useLiveConversationMetrics(
  enabled: boolean = true,
): MetricsState {
  // Per-field selectors rather than a bare `useMetricsStore()`: subscribing to
  // the whole store re-runs this hook on any store write, including the action
  // identity, and matches how `useContextWindowUsage` reads the same store.
  const storeCost = useMetricsStore((state) => state.cost);
  const storeMaxBudgetPerTask = useMetricsStore(
    (state) => state.max_budget_per_task,
  );
  const storeUsage = useMetricsStore((state) => state.usage);
  const { data: conversation } = useActiveConversation();

  const { data: conversationMetrics } = useConversationMetrics(
    conversation?.id,
    conversation?.conversation_url,
    conversation?.session_api_key,
    enabled,
  );

  return useMemo(() => {
    // Live store first: real-time and (post-reset) always the active
    // conversation's data. Prefer the store's `usage` whenever it is
    // populated; a partial WebSocket stats event can carry `cost` from a
    // usage id whose `accumulated_token_usage` is null, and that case
    // must still consult the REST snapshot for token fields like
    // `cache_read_tokens` (issue #16835). When the store has no usage yet
    // but REST has loaded, fall through to the snapshot for those fields.
    if (storeUsage !== null) {
      return {
        cost: storeCost,
        max_budget_per_task: storeMaxBudgetPerTask,
        usage: storeUsage,
      };
    }

    if (conversationMetrics) {
      return {
        cost: storeCost ?? conversationMetrics.accumulated_cost,
        max_budget_per_task:
          storeMaxBudgetPerTask ?? conversationMetrics.max_budget_per_task,
        usage: formatUsage(conversationMetrics.accumulated_token_usage),
      };
    }

    return {
      cost: storeCost,
      max_budget_per_task: storeMaxBudgetPerTask,
      usage: null,
    };
  }, [conversationMetrics, storeCost, storeMaxBudgetPerTask, storeUsage]);
}

function formatUsage(
  accumulatedTokenUsage:
    | MetricsSnapshot["accumulated_token_usage"]
    | null
    | undefined,
) {
  if (!accumulatedTokenUsage) {
    return null;
  }
  return {
    prompt_tokens: accumulatedTokenUsage.prompt_tokens ?? 0,
    completion_tokens: accumulatedTokenUsage.completion_tokens ?? 0,
    cache_read_tokens: accumulatedTokenUsage.cache_read_tokens ?? 0,
    cache_write_tokens: accumulatedTokenUsage.cache_write_tokens ?? 0,
    context_window: accumulatedTokenUsage.context_window ?? 0,
    per_turn_token: accumulatedTokenUsage.per_turn_token ?? 0,
  };
}
