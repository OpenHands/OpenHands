import type {
  V1MetricsSnapshot,
  V1RuntimeConversationInfo,
  V1TokenUsage,
} from "#/api/conversation-service/v1-conversation-service.types";

/**
 * TypeScript equivalent of the get_combined_metrics method from the Python SDK
 * Combines metrics from all LLM usage IDs in the conversation stats
 */
export function getCombinedMetrics(
  conversationInfo: V1RuntimeConversationInfo,
): V1MetricsSnapshot {
  const { stats } = conversationInfo;

  if (!stats?.usage_to_metrics) {
    return {
      accumulated_cost: 0,
      max_budget_per_task: null,
      accumulated_token_usage: null,
    };
  }

  let totalCost = 0;
  let maxBudgetPerTask: number | null = null;
  let combinedTokenUsage: V1TokenUsage | null = null;

  // Iterate through all metrics and combine them
  for (const metrics of Object.values(stats.usage_to_metrics)) {
    // Add up costs
    totalCost += metrics.accumulated_cost;

    // Keep the max budget per task if any is set
    if (maxBudgetPerTask === null && metrics.max_budget_per_task !== null) {
      maxBudgetPerTask = metrics.max_budget_per_task;
    }

    // Combine token usage
    if (metrics.accumulated_token_usage) {
      if (combinedTokenUsage === null) {
        combinedTokenUsage = { ...metrics.accumulated_token_usage };
      } else {
        combinedTokenUsage = {
          prompt_tokens:
            combinedTokenUsage.prompt_tokens +
            metrics.accumulated_token_usage.prompt_tokens,
          completion_tokens:
            combinedTokenUsage.completion_tokens +
            metrics.accumulated_token_usage.completion_tokens,
          cache_read_tokens:
            combinedTokenUsage.cache_read_tokens +
            metrics.accumulated_token_usage.cache_read_tokens,
          cache_write_tokens:
            combinedTokenUsage.cache_write_tokens +
            metrics.accumulated_token_usage.cache_write_tokens,
          context_window: Math.max(
            combinedTokenUsage.context_window,
            metrics.accumulated_token_usage.context_window,
          ),
          per_turn_token: metrics.accumulated_token_usage.per_turn_token, // Use the latest per_turn_token
        };
      }
    }
  }

  return {
    accumulated_cost: totalCost,
    max_budget_per_task: maxBudgetPerTask,
    accumulated_token_usage: combinedTokenUsage,
  };
}
