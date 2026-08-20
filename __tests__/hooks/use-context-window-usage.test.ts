import { renderHook } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import useMetricsStore, { type MetricsState } from "#/stores/metrics-store";
import type { TokenUsage } from "#/api/conversation-service/agent-server-conversation-service.types";

const useActiveConversationMock = vi.fn();
const useConversationMetricsMock = vi.fn();

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));

vi.mock("#/hooks/query/use-conversation-metrics", () => ({
  useConversationMetrics: (...args: unknown[]) =>
    useConversationMetricsMock(...args),
}));

import { useContextWindowUsage } from "#/hooks/use-context-window-usage";

const CONVERSATION = {
  id: "conv-1",
  conversation_url: "http://localhost:54928/api/conversations/conv-1",
  session_api_key: "sess-key",
};

const storeUsage = (
  contextWindow: number,
  perTurnToken: number,
): NonNullable<MetricsState["usage"]> => ({
  prompt_tokens: 10,
  completion_tokens: 20,
  cache_read_tokens: 0,
  cache_write_tokens: 0,
  context_window: contextWindow,
  per_turn_token: perTurnToken,
});

// The wire can deliver nulls for any token field even though the SDK type
// says number; the hook's `?? 0` coercion is what these tests pin.
const restUsage = (
  contextWindow: number | null,
  perTurnToken: number | null,
): TokenUsage =>
  ({
    prompt_tokens: 10,
    completion_tokens: 20,
    cache_read_tokens: 0,
    cache_write_tokens: 0,
    context_window: contextWindow,
    per_turn_token: perTurnToken,
  }) as unknown as TokenUsage;

describe("useContextWindowUsage", () => {
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

  it("prefers the live WS store when it reports a usable context window", () => {
    useMetricsStore.setState({ usage: storeUsage(200_000, 1234) });
    useConversationMetricsMock.mockReturnValue({
      data: { accumulated_token_usage: restUsage(128_000, 500) },
    });

    const { result } = renderHook(() => useContextWindowUsage());

    expect(result.current).toEqual({
      perTurnToken: 1234,
      contextWindow: 200_000,
    });
  });

  it("falls back to the REST snapshot when the store has no usage", () => {
    useConversationMetricsMock.mockReturnValue({
      data: { accumulated_token_usage: restUsage(128_000, 500) },
    });

    const { result } = renderHook(() => useContextWindowUsage());

    expect(result.current).toEqual({
      perTurnToken: 500,
      contextWindow: 128_000,
    });
  });

  it("falls through to REST when the store reports a zero context window", () => {
    useMetricsStore.setState({ usage: storeUsage(0, 999) });
    useConversationMetricsMock.mockReturnValue({
      data: { accumulated_token_usage: restUsage(128_000, 500) },
    });

    const { result } = renderHook(() => useContextWindowUsage());

    expect(result.current).toEqual({
      perTurnToken: 500,
      contextWindow: 128_000,
    });
  });

  it("returns null when no source reports a positive context window", () => {
    // The divide-by-zero guard: a model that does not report a context
    // window must render no meter at all.
    useConversationMetricsMock.mockReturnValue({
      data: { accumulated_token_usage: restUsage(0, 500) },
    });

    const { result } = renderHook(() => useContextWindowUsage());

    expect(result.current).toBeNull();
  });

  it("coerces a null per_turn_token from the wire to zero", () => {
    useConversationMetricsMock.mockReturnValue({
      data: { accumulated_token_usage: restUsage(128_000, null) },
    });

    const { result } = renderHook(() => useContextWindowUsage());

    expect(result.current).toEqual({
      perTurnToken: 0,
      contextWindow: 128_000,
    });
  });

  it("polls the REST snapshot only while a conversation is active", () => {
    renderHook(() => useContextWindowUsage());
    expect(useConversationMetricsMock).toHaveBeenCalledWith(
      "conv-1",
      CONVERSATION.conversation_url,
      "sess-key",
      true,
    );

    useConversationMetricsMock.mockClear();
    useActiveConversationMock.mockReturnValue({ data: undefined });
    renderHook(() => useContextWindowUsage());
    expect(useConversationMetricsMock).toHaveBeenCalledWith(
      undefined,
      undefined,
      undefined,
      false,
    );
  });
});
