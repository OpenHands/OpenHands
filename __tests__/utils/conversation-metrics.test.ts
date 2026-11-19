import { describe, expect, it } from "vitest";
import { MetricsSnapshot } from "#/api/conversation-service/agent-server-conversation-service.types";
import {
  buildUsageBreakdown,
  combineUsageMetrics,
} from "#/utils/conversation-metrics";

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

  it("combines cost and tokens and attributes them per model", () => {
    const result = snapshotFrom({
      default: metrics("gpt-4o", 1.25, usage({ prompt_tokens: 100 })),
      condenser: metrics("gpt-4o-mini", 0.5, usage({ prompt_tokens: 20 })),
    });

    expect(result.accumulated_cost).toBe(1.75);
    expect(result.accumulated_token_usage?.prompt_tokens).toBe(120);
    expect(result.usage_breakdown).toEqual([
      {
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
        modelName: "gpt-4o-mini",
        accumulatedCost: 0.5,
        usage: {
          prompt_tokens: 20,
          completion_tokens: 50,
          cache_read_tokens: 10,
          cache_write_tokens: 5,
        },
      },
    ]);
    const breakdownTotal = result.usage_breakdown!.reduce(
      (sum, entry) => sum + entry.accumulatedCost,
      0,
    );
    expect(breakdownTotal).toBe(result.accumulated_cost);
  });
});

describe("buildUsageBreakdown", () => {
  it("sorts models by accumulated cost descending", () => {
    const result = buildUsageBreakdown({
      default: metrics("gpt-4o", 1.0),
      condenser: metrics("gpt-4o-mini", 2.0),
    });
    expect(result.map((entry) => entry.modelName)).toEqual([
      "gpt-4o-mini",
      "gpt-4o",
    ]);
  });

  it("sums usage ids that share a model into one entry", () => {
    const result = buildUsageBreakdown({
      default: metrics("gpt-4o", 1.0, usage({ prompt_tokens: 100 })),
      condenser: metrics("gpt-4o", 0.25, usage({ prompt_tokens: 40 })),
    });
    expect(result).toHaveLength(1);
    expect(result[0].accumulatedCost).toBe(1.25);
    expect(result[0].usage.prompt_tokens).toBe(140);
  });

  it("drops usage ids that never ran and keeps unpriced ones that did", () => {
    const zero = usage({ prompt_tokens: 0, completion_tokens: 0 });
    const result = buildUsageBreakdown({
      default: metrics("acme/coder", 0, usage({ prompt_tokens: 1_000 })),
      condenser: metrics("gpt-4o-mini", 0, zero),
    });
    expect(result.map((entry) => entry.modelName)).toEqual(["acme/coder"]);
  });

  it("falls back to the usage id when the model name is empty", () => {
    const result = buildUsageBreakdown({ agent: metrics("", 0.1) });
    expect(result[0].modelName).toBe("agent");
  });
});
