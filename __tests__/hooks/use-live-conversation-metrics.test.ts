import { renderHook } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import useMetricsStore, { type MetricsState } from "#/stores/metrics-store";
import type {
  MetricsSnapshot,
  TokenUsage,
} from "#/api/conversation-service/agent-server-conversation-service.types";

const useActiveConversationMock = vi.fn();
const useConversationMetricsMock = vi.fn();

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));

vi.mock("#/hooks/query/use-conversation-metrics", () => ({
  useConversationMetrics: (...args: unknown[]) =>
    useConversationMetricsMock(...args),
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

const snapshot = (
  overrides: Partial<MetricsSnapshot> = {},
): MetricsSnapshot => ({
  accumulated_cost: 1.23,
  max_budget_per_task: 5,
  accumulated_token_usage: restUsage(),
  ...overrides,
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

describe("useLiveConversationMetrics", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useMetricsStore.setState({
      cost: null,
      max_budget_per_task: null,
      usage: null,
    });
    useActiveConversationMock.mockReturnValue({ data: CONVERSATION });
    useConversationMetricsMock.mockReturnValue({ data: undefined });
  });

  it("prefers the live store over the REST snapshot", () => {
    useMetricsStore.setState(storeMetrics);
    useConversationMetricsMock.mockReturnValue({ data: snapshot() });

    const { result } = renderHook(() => useLiveConversationMetrics());

    expect(result.current).toEqual(storeMetrics);
  });

  it("coerces null token fields from the wire to zero", () => {
    useConversationMetricsMock.mockReturnValue({
      data: snapshot({
        accumulated_token_usage: restUsage({
          completion_tokens: null,
          context_window: null,
          per_turn_token: null,
        }),
      }),
    });

    const { result } = renderHook(() => useLiveConversationMetrics());

    expect(result.current.usage).toEqual({
      prompt_tokens: 10,
      completion_tokens: 0,
      cache_read_tokens: 1,
      cache_write_tokens: 2,
      context_window: 0,
      per_turn_token: 0,
    });
  });

  it("maps a missing accumulated_token_usage to null usage", () => {
    useConversationMetricsMock.mockReturnValue({
      data: snapshot({ accumulated_token_usage: null }),
    });

    const { result } = renderHook(() => useLiveConversationMetrics());

    expect(result.current).toEqual({
      cost: 1.23,
      max_budget_per_task: 5,
      usage: null,
    });
  });

  it("falls back to the REST snapshot while the store is empty", () => {
    useConversationMetricsMock.mockReturnValue({ data: snapshot() });

    const { result } = renderHook(() => useLiveConversationMetrics());

    expect(result.current).toEqual({
      cost: 1.23,
      max_budget_per_task: 5,
      usage: {
        prompt_tokens: 10,
        completion_tokens: 20,
        cache_read_tokens: 1,
        cache_write_tokens: 2,
        context_window: 128_000,
        per_turn_token: 500,
      },
    });
  });

  it("threads the enabled flag through to the polling query", () => {
    renderHook(() => useLiveConversationMetrics(false));
    expect(useConversationMetricsMock).toHaveBeenCalledWith(
      "conv-1",
      CONVERSATION.conversation_url,
      "sess-key",
      false,
    );

    useConversationMetricsMock.mockClear();
    renderHook(() => useLiveConversationMetrics());
    expect(useConversationMetricsMock).toHaveBeenCalledWith(
      "conv-1",
      CONVERSATION.conversation_url,
      "sess-key",
      true,
    );
  });
});
