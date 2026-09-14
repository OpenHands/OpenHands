/* Integration: useCompactContextAction end-to-end through real stores. */
import { act, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import useMetricsStore from "#/stores/metrics-store";
import { useEventStore } from "#/stores/use-event-store";
import type { OpenHandsEvent } from "#/types/agent-server/core";

const mutateMock = vi.fn();
vi.mock("#/hooks/mutation/use-condense-conversation", () => ({
  useCondenseConversation: () => ({ mutate: mutateMock, isPending: false }),
}));
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: { id: "c1", conversation_url: null, session_api_key: null },
  }),
}));
vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => ({ curAgentState: "FINISHED" }),
}));
const successToast = vi.fn();
const errorToast = vi.fn();
vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: (...args: unknown[]) => successToast(...args),
  displayErrorToast: (...args: unknown[]) => errorToast(...args),
}));

import { useCompactContextAction } from "#/hooks/use-compact-context-action";
import { I18nKey } from "#/i18n/declaration";

function condensationEvent(id: string): OpenHandsEvent {
  return {
    id,
    timestamp: new Date().toISOString(),
    source: "environment",
    kind: "Condensation",
    forgotten_event_ids: ["a"],
  } as OpenHandsEvent;
}

const usage = (perTurn: number) => ({
  prompt_tokens: perTurn,
  completion_tokens: 0,
  cache_read_tokens: 0,
  cache_write_tokens: 0,
  context_window: 262144,
  per_turn_token: perTurn,
});

function wrapper({ children }: { children: React.ReactNode }) {
  return React.createElement(
    QueryClientProvider,
    { client: new QueryClient() },
    children,
  );
}

describe("useCompactContextAction (integration)", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    useEventStore.getState().clearEvents();
    useMetricsStore.getState().setMetrics({
      cost: null,
      max_budget_per_task: null,
      usage: usage(50_000),
    });
    mutateMock.mockImplementation((_vars, opts) => opts.onSuccess());
    successToast.mockClear();
    errorToast.mockClear();
  });

  afterEach(() => {
    vi.useRealTimers();
    useEventStore.getState().clearEvents();
  });

  it("toasts counts when condensation lands and metrics drop", () => {
    const { result } = renderHook(() => useCompactContextAction(50_000), {
      wrapper,
    });

    act(() => result.current.handleCompact());
    expect(successToast).toHaveBeenCalledWith(
      I18nKey.CONVERSATION$COMPACT_CONTEXT_STARTED,
    );

    act(() => {
      useEventStore.getState().addEvent(condensationEvent("cond-x"));
      useMetricsStore.getState().setMetrics({
        cost: null,
        max_budget_per_task: null,
        usage: usage(20_000),
      });
    });

    expect(successToast).toHaveBeenCalledWith(
      I18nKey.CONVERSATION$COMPACT_CONTEXT_COMPLETE,
    );
    expect(errorToast).not.toHaveBeenCalled();
  });

  it("toasts error when no condensation arrives in time", () => {
    const { result } = renderHook(() => useCompactContextAction(50_000), {
      wrapper,
    });

    act(() => result.current.handleCompact());
    act(() => {
      vi.advanceTimersByTime(90_000);
    });

    expect(errorToast).toHaveBeenCalledWith(
      I18nKey.CONVERSATION$COMPACT_CONTEXT_FAILED,
    );
  });
});
