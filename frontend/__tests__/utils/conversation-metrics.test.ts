import { describe, expect, it } from "vitest";

import { combineUsageMetrics } from "#/utils/conversation-metrics";

const usage = (
  prompt: number,
  completion: number,
  overrides: Partial<{
    cache_read_tokens: number;
    cache_write_tokens: number;
    context_window: number;
    per_turn_token: number;
  }> = {},
) => ({
  prompt_tokens: prompt,
  completion_tokens: completion,
  cache_read_tokens: 0,
  cache_write_tokens: 0,
  context_window: 0,
  per_turn_token: 0,
  ...overrides,
});

describe("combineUsageMetrics", () => {
  // A mid-conversation LLM switch accrues under "profile:*" and ACP under
  // "acp-managed"; reading only the "agent" bucket under-reports.
  it("sums cost and tokens across every usage bucket", () => {
    const combined = combineUsageMetrics({
      agent: {
        accumulated_cost: 0.0824,
        max_budget_per_task: null,
        accumulated_token_usage: usage(100, 10, { context_window: 200000 }),
      },
      condenser: {
        accumulated_cost: 0,
        max_budget_per_task: null,
        accumulated_token_usage: null,
      },
      "profile:opus-repro:abc123": {
        accumulated_cost: 0.1741,
        max_budget_per_task: null,
        accumulated_token_usage: usage(200, 20, { per_turn_token: 42 }),
      },
    });

    expect(combined.accumulated_cost).toBeCloseTo(0.2565, 10);
    expect(combined.accumulated_token_usage?.prompt_tokens).toBe(300);
    expect(combined.accumulated_token_usage?.completion_tokens).toBe(30);
    expect(combined.accumulated_token_usage?.context_window).toBe(200000);
  });

  it("keeps the first non-null max budget", () => {
    const combined = combineUsageMetrics({
      agent: {
        accumulated_cost: 0,
        max_budget_per_task: 10,
        accumulated_token_usage: null,
      },
      "acp-managed": {
        accumulated_cost: 0.5,
        max_budget_per_task: null,
        accumulated_token_usage: usage(50, 5),
      },
    });

    expect(combined.max_budget_per_task).toBe(10);
    expect(combined.accumulated_cost).toBeCloseTo(0.5, 10);
    expect(combined.accumulated_token_usage?.prompt_tokens).toBe(50);
  });
});
