import { afterEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { useConversationMetrics } from "#/hooks/query/use-conversation-metrics";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

const runtimeInfo = {
  id: "conv-abc",
  title: "Test",
  metrics: null,
  created_at: "2026-04-16T00:00:00Z",
  updated_at: "2026-04-16T00:00:00Z",
  status: ExecutionStatus.IDLE,
  stats: {
    usage_to_metrics: {
      agent: {
        model_name: "test-model",
        accumulated_cost: 2.5,
        max_budget_per_task: null,
        accumulated_token_usage: null,
        costs: [],
        response_latencies: [],
        token_usages: [],
      },
    },
  },
};

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useConversationMetrics", () => {
  it("fires the query when sessionApiKey is null (local backends without auth)", async () => {
    // Arrange
    const spy = vi
      .spyOn(AgentServerConversationService, "getRuntimeConversation")
      .mockResolvedValue(runtimeInfo);

    // Act
    const { result } = renderHook(
      () =>
        useConversationMetrics(
          "conv-abc",
          "http://localhost:8888/api/conversations/conv-abc",
          null,
          true,
        ),
      { wrapper: makeWrapper() },
    );

    // Assert
    await waitFor(() => {
      expect(result.current.data?.accumulated_cost).toBe(2.5);
    });
    expect(spy).toHaveBeenCalledWith(
      "conv-abc",
      "http://localhost:8888/api/conversations/conv-abc",
      null,
    );
  });

  it("preserves an active LLM's per_turn_token when combined with a dormant condenser entry", async () => {
    // Arrange
    const multiLlmRuntimeInfo = {
      id: "conv-multi",
      title: "Test",
      metrics: null,
      created_at: "2026-04-16T00:00:00Z",
      updated_at: "2026-04-16T00:00:00Z",
      status: ExecutionStatus.IDLE,
      stats: {
        usage_to_metrics: {
          agent: {
            model_name: "test-model",
            accumulated_cost: 0.07,
            max_budget_per_task: null,
            accumulated_token_usage: {
              prompt_tokens: 37802,
              completion_tokens: 1024,
              cache_read_tokens: 24686,
              cache_write_tokens: 12764,
              context_window: 1_000_000,
              per_turn_token: 38826,
            },
            costs: [],
            response_latencies: [],
            token_usages: [],
          },
          condenser: {
            model_name: "test-model",
            accumulated_cost: 0,
            max_budget_per_task: null,
            accumulated_token_usage: {
              prompt_tokens: 0,
              completion_tokens: 0,
              cache_read_tokens: 0,
              cache_write_tokens: 0,
              context_window: 0,
              per_turn_token: 0,
            },
            costs: [],
            response_latencies: [],
            token_usages: [],
          },
        },
      },
    };
    vi.spyOn(
      AgentServerConversationService,
      "getRuntimeConversation",
    ).mockResolvedValue(multiLlmRuntimeInfo);

    // Act
    const { result } = renderHook(
      () =>
        useConversationMetrics(
          "conv-multi",
          "http://localhost:8888/api/conversations/conv-multi",
          null,
          true,
        ),
      { wrapper: makeWrapper() },
    );

    // Assert
    await waitFor(() => {
      expect(
        result.current.data?.accumulated_token_usage?.per_turn_token,
      ).toBe(38826);
    });
  });
});
