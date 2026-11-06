import { describe, expect, it } from "vitest";
import { MetricsSnapshot } from "#/api/conversation-service/agent-server-conversation-service.types";
import { combineUsageMetrics } from "#/utils/conversation-metrics";

const usage = (overrides: Record<string, number> = {}) => ({
  prompt_tokens: 100,
  completion_tokens: 50,
  cache_read_tokens: 10,
  cache_write_tokens: 5,
  context_window: 128_000,
  per_turn_token: 150,
  ...overrides,
});

const metrics = (modelName: string, cost: number, tokenUsage = usage()) => ({
  model_name: modelName,
  accumulated_cost: cost,
  max_budget_per_task: null,
  accumulated_token_usage: tokenUsage,
  costs: [],
  response_latencies: [],
  token_usages: [],
});

const snapshotFrom = (
  usageToMetrics: Record<string, ReturnType<typeof metrics>>,
): MetricsSnapshot =>
  combineUsageMetrics({ usage_to_metrics: usageToMetrics } as never);

describe("combineUsageMetrics", () => {
  it("returns empty snapshot when stats are missing", () => {
    expect(combineUsageMetrics(null)).toEqual({
      accumulated_cost: 0,
      max_budget_per_task: null,
      accumulated_token_usage: null,
    });
  });

  it("combines cost and tokens across usage ids", () => {
    const result = snapshotFrom({
      default: metrics("gpt-4o", 1.25, usage({ prompt_tokens: 100 })),
      condenser: metrics("gpt-4o-mini", 0.5, usage({ prompt_tokens: 20 })),
    });

    expect(result.accumulated_cost).toBe(1.75);
    expect(result.accumulated_token_usage?.prompt_tokens).toBe(120);
  });

  it("keeps a per-model breakdown with usage ids and costs", () => {
    const result = snapshotFrom({
      default: metrics("gpt-4o", 1.25, usage()),
      condenser: metrics("gpt-4o-mini", 0.5, usage()),
    });

    expect(result.usage_breakdown).toEqual([
      {
        usageId: "default",
        modelName: "gpt-4o",
        accumulatedCost: 1.25,
        usage: {
          prompt_tokens: 100,
          completion_tokens: 50,
          cache_read_tokens: 10,
          cache_write_tokens: 5,
        },
      },
      {
        usageId: "condenser",
        modelName: "gpt-4o-mini",
        accumulatedCost: 0.5,
        usage: {
          prompt_tokens: 100,
          completion_tokens: 50,
          cache_read_tokens: 10,
          cache_write_tokens: 5,
        },
      },
    ]);
  });

  it("sorts the breakdown by accumulated cost descending", () => {
    const result = snapshotFrom({
      default: metrics("gpt-4o", 1.0),
      condenser: metrics("gpt-4o-mini", 2.0),
    });

    expect(
      result.usage_breakdown?.map((entry) => entry.accumulatedCost),
    ).toEqual([2.0, 1.0]);
  });
});
