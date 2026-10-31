import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import useMetricsStore, { type MetricsState } from "#/stores/metrics-store";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type {
  RuntimeMetrics,
  RuntimeConversationInfo,
  TokenUsage,
} from "#/api/conversation-service/agent-server-conversation-service.types";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

const useActiveConversationMock = vi.fn();

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));

import { useLiveConversationMetrics } from "#/hooks/use-live-conversation-metrics";

const CONVERSATION = {
  id: "conv-1",
  conversation_url: "http://localhost:54928/api/conversations/conv-1",
  session_api_key: "sess-key",
};

// The wire can deliver nulls for any token field even though the SDK type
// says number; the hook's `?? 0` coercion is what these tests pin.
const restUsage = (overrides: Record<string, number | null> = {}): TokenUsage =>
  ({
    prompt_tokens: 10,
    completion_tokens: 20,
    cache_read_tokens: 1,
    cache_write_tokens: 2,
    context_window: 128_000,
    per_turn_token: 500,
    ...overrides,
  }) as unknown as TokenUsage;

const runtimeInfo = (
  metrics: Partial<RuntimeMetrics> = {},
  maxBudget: number | null = 5,
): RuntimeConversationInfo => ({
  id: CONVERSATION.id,
  title: "Budget conversation",
  created_at: "2026-10-26T00:00:00Z",
  updated_at: "2026-10-26T00:00:00Z",
  status: ExecutionStatus.IDLE,
  available_models: [],
  metrics: null,
  max_budget_per_run: maxBudget,
  stats: {
    usage_to_metrics: {
      agent: {
        model_name: "test-model",
        accumulated_cost: 1.23,
        max_budget_per_task: 10,
        accumulated_token_usage: restUsage(),
        costs: [],
        response_latencies: [],
        token_usages: [],
        ...metrics,
      },
    },
  },
});

const storeMetrics: MetricsState = {
  cost: 9.99,
  max_budget_per_task: 10,
  usage: {
    prompt_tokens: 1,
    completion_tokens: 2,
    cache_read_tokens: 3,
    cache_write_tokens: 4,
    context_window: 64_000,
    per_turn_token: 42,
  },
};

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client }, children);
}

describe("useLiveConversationMetrics", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useMetricsStore.getState().resetMetrics();
    useActiveConversationMock.mockReturnValue({ data: CONVERSATION });
    vi.spyOn(
      AgentServerConversationService,
      "getRuntimeConversation",
    ).mockResolvedValue(runtimeInfo());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("retains the conversation cap when live cost replaces the REST snapshot", async () => {
    const { result } = renderHook(() => useLiveConversationMetrics(), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.cost).toBe(1.23));
    expect(result.current.maxBudget).toBe(5);

    act(() =>
      useMetricsStore.getState().setMetrics(storeMetrics, CONVERSATION.id),
    );

    expect(result.current).toEqual({
      cost: 9.99,
      maxBudget: 5,
      usage: storeMetrics.usage,
    });
  });

  it("coerces null token fields from the wire to zero", async () => {
    vi.mocked(
      AgentServerConversationService.getRuntimeConversation,
    ).mockResolvedValue(
      runtimeInfo({
        accumulated_token_usage: restUsage({
          completion_tokens: null,
          context_window: null,
          per_turn_token: null,
        }),
      }),
    );
    const { result } = renderHook(() => useLiveConversationMetrics(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() =>
      expect(result.current.usage).toEqual({
        prompt_tokens: 10,
        completion_tokens: 0,
        cache_read_tokens: 1,
        cache_write_tokens: 2,
        context_window: 0,
        per_turn_token: 0,
      }),
    );
  });

  it("maps missing token usage to null usage", async () => {
    vi.mocked(
      AgentServerConversationService.getRuntimeConversation,
    ).mockResolvedValue(runtimeInfo({ accumulated_token_usage: null }));
    const { result } = renderHook(() => useLiveConversationMetrics(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() =>
      expect(result.current).toEqual({
        cost: 1.23,
        maxBudget: 5,
        usage: null,
      }),
    );
  });

  it("drops the previous cap when switching to an unbounded conversation", async () => {
    vi.mocked(
      AgentServerConversationService.getRuntimeConversation,
    ).mockImplementation(async (id) =>
      id === CONVERSATION.id
        ? runtimeInfo()
        : runtimeInfo({ accumulated_cost: 2 }, null),
    );
    const { result, rerender } = renderHook(
      () => useLiveConversationMetrics(),
      {
        wrapper: makeWrapper(),
      },
    );
    await waitFor(() => expect(result.current.maxBudget).toBe(5));

    useActiveConversationMock.mockReturnValue({
      data: { ...CONVERSATION, id: "conv-2", conversation_url: null },
    });
    rerender();

    expect(result.current.maxBudget).toBeNull();
    await waitFor(() => expect(result.current.cost).toBe(2));
    expect(result.current.maxBudget).toBeNull();
  });

  it("pauses the runtime query until enabled", async () => {
    const { result, rerender } = renderHook(
      ({ enabled }) => useLiveConversationMetrics(enabled),
      { wrapper: makeWrapper(), initialProps: { enabled: false } },
    );
    expect(
      AgentServerConversationService.getRuntimeConversation,
    ).not.toHaveBeenCalled();

    rerender({ enabled: true });

    await waitFor(() => expect(result.current.cost).toBe(1.23));
  });
});
