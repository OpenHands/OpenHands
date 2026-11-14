import type {
  MetricsSnapshot,
  RuntimeConversationInfo,
  RuntimeConversationStats,
  TokenUsage,
  UsageBreakdownEntry,
} from "#/api/conversation-service/agent-server-conversation-service.types";

interface UsageMetricsLike {
  model_name: string;
  accumulated_cost: number;
  accumulated_token_usage: UsageBreakdownEntry["usage"] | null;
}

/**
 * Per-model cost and tokens, costliest first, so the combined total can be
 * attributed. Usage ids sharing a model (e.g. agent + condenser) are summed and
 * unused ones dropped. Shared by the REST snapshot and the live stats event.
 */
export function buildUsageBreakdown(
  usageToMetrics: Record<string, UsageMetricsLike>,
): UsageBreakdownEntry[] {
  const byModel = new Map<string, UsageBreakdownEntry>();
  for (const [usageId, metrics] of Object.entries(usageToMetrics)) {
    const modelName = metrics.model_name || usageId;
    const entry = byModel.get(modelName) ?? {
      modelName,
      accumulatedCost: 0,
      usage: {
        prompt_tokens: 0,
        completion_tokens: 0,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
      },
    };
    entry.accumulatedCost += metrics.accumulated_cost;
    const tokens = metrics.accumulated_token_usage;
    if (tokens) {
      entry.usage.prompt_tokens += tokens.prompt_tokens;
      entry.usage.completion_tokens += tokens.completion_tokens;
      entry.usage.cache_read_tokens += tokens.cache_read_tokens;
      entry.usage.cache_write_tokens += tokens.cache_write_tokens;
    }
    byModel.set(modelName, entry);
  }
  return [...byModel.values()]
    .filter(
      (entry) =>
        entry.accumulatedCost > 0 ||
        entry.usage.prompt_tokens + entry.usage.completion_tokens > 0,
    )
    .sort((a, b) => b.accumulatedCost - a.accumulatedCost);
}

/**
 * TypeScript equivalent of the get_combined_metrics method from the Python SDK
 * Combines metrics from all LLM usage IDs in a conversation's stats
 */
export function combineUsageMetrics(
  stats: RuntimeConversationStats | null | undefined,
): MetricsSnapshot {
  if (!stats?.usage_to_metrics) {
    return {
      accumulated_cost: 0,
      max_budget_per_task: null,
      accumulated_token_usage: null,
    };
  }

  let totalCost = 0;
  let maxBudgetPerTask: number | null = null;
  let combinedTokenUsage: TokenUsage | null = null;

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
          per_turn_token: Math.max(
            combinedTokenUsage.per_turn_token,
            metrics.accumulated_token_usage.per_turn_token,
          ),
        };
      }
    }
  }

  return {
    accumulated_cost: totalCost,
    max_budget_per_task: maxBudgetPerTask,
    accumulated_token_usage: combinedTokenUsage,
    usage_breakdown: buildUsageBreakdown(stats.usage_to_metrics),
  };
}

export function getCombinedMetrics(
  conversationInfo: RuntimeConversationInfo,
): MetricsSnapshot {
  return combineUsageMetrics(conversationInfo.stats);
}
