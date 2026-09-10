import { describe, it, expect } from "vitest";
import { combineUsageMetrics } from "#/utils/conversation-metrics";
import type { RuntimeConversationStats } from "#/api/conversation-service/agent-server-conversation-service.types";

function usage(perTurnToken: number, promptTokens: number) {
  return {
    prompt_tokens: promptTokens,
    completion_tokens: 10,
    cache_read_tokens: 0,
    cache_write_tokens: 0,
    context_window: 1_000_000,
    per_turn_token: perTurnToken,
  };
}

function metricsEntry(perTurnToken: number, promptTokens: number) {
  return {
    accumulated_cost: 0.01,
    max_budget_per_task: null,
    accumulated_token_usage: usage(perTurnToken, promptTokens),
  };
}

describe("combineUsageMetrics", () => {
  it("sums token totals across usage entries", () => {
    const stats = {
      usage_to_metrics: {
        default: metricsEntry(2000, 100),
        condenser: metricsEntry(8000, 400),
      },
    } as unknown as RuntimeConversationStats;
    const combined = combineUsageMetrics(stats);
    expect(combined.accumulated_token_usage?.prompt_tokens).toBe(500);
  });

  it("takes per_turn_token from the primary usage, not the max across services", () => {
    // Post-compaction: the agent's context dropped to 2000 tokens, but the
    // condenser's own last call read the full 8000-token history. A max across
    // services pins the context meter (and the compaction hook's drop check)
    // to the condenser's stale value.
    const stats = {
      usage_to_metrics: {
        default: metricsEntry(2000, 100),
        condenser: metricsEntry(8000, 400),
      },
    } as unknown as RuntimeConversationStats;
    const combined = combineUsageMetrics(stats);
    expect(combined.accumulated_token_usage?.per_turn_token).toBe(2000);
  });
});
