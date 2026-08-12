import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import useMetricsStore from "#/stores/metrics-store";
import { useEventStore } from "#/stores/use-event-store";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import {
  buildContextCompactionResult,
  useAwaitContextCompaction,
} from "./use-await-context-compaction";

function condensationEvent(id: string): OpenHandsEvent {
  return {
    id,
    timestamp: new Date().toISOString(),
    source: "environment",
    kind: "Condensation",
    forgotten_event_ids: ["a", "b"],
  } as OpenHandsEvent;
}

describe("buildContextCompactionResult", () => {
  it("computes tokens saved without going negative", () => {
    expect(buildContextCompactionResult(1000, 400, "compacted")).toEqual({
      beforeToken: 1000,
      afterToken: 400,
      savedToken: 600,
      outcome: "compacted",
    });
    expect(buildContextCompactionResult(400, 500, "no_change").savedToken).toBe(
      0,
    );
  });
});

describe("useAwaitContextCompaction", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    useEventStore.getState().clearEvents();
    useMetricsStore.getState().setMetrics({
      cost: null,
      max_budget_per_task: null,
      usage: {
        prompt_tokens: 0,
        completion_tokens: 0,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        context_window: 100_000,
        per_turn_token: 10_000,
      },
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    useEventStore.getState().clearEvents();
  });

  it("reports tokens freed once condensation lands and metrics drop", () => {
    const onComplete = vi.fn();

    renderHook(() =>
      useAwaitContextCompaction({
        beforeToken: 10_000,
        onComplete,
      }),
    );

    act(() => {
      useEventStore.getState().addEvent(condensationEvent("cond-1"));
      useMetricsStore.getState().setMetrics({
        cost: null,
        max_budget_per_task: null,
        usage: {
          prompt_tokens: 0,
          completion_tokens: 0,
          cache_read_tokens: 0,
          cache_write_tokens: 0,
          context_window: 100_000,
          per_turn_token: 4_000,
        },
      });
    });

    expect(onComplete).toHaveBeenCalledWith({
      beforeToken: 10_000,
      afterToken: 4_000,
      savedToken: 6_000,
      outcome: "compacted",
    });
  });

  it("finishes after the settle window when metrics do not drop", () => {
    const onComplete = vi.fn();

    renderHook(() =>
      useAwaitContextCompaction({
        beforeToken: 10_000,
        onComplete,
      }),
    );

    act(() => {
      useEventStore.getState().addEvent(condensationEvent("cond-2"));
    });
    expect(onComplete).not.toHaveBeenCalled();

    act(() => {
      vi.advanceTimersByTime(2_500);
    });

    expect(onComplete).toHaveBeenCalledWith({
      beforeToken: 10_000,
      afterToken: 10_000,
      savedToken: 0,
      outcome: "no_change",
    });
  });

  it("reports timeout when no condensation event arrives in time", () => {
    const onComplete = vi.fn();

    renderHook(() =>
      useAwaitContextCompaction({
        beforeToken: 10_000,
        onComplete,
      }),
    );

    act(() => {
      vi.advanceTimersByTime(90_000);
    });

    expect(onComplete).toHaveBeenCalledWith({
      beforeToken: 10_000,
      afterToken: 10_000,
      savedToken: 0,
      outcome: "timeout",
    });
  });

  it("detects a condensation that landed before the post-ack effect started", () => {
    // The condense POST can return only after the server already emitted its
    // Condensation event. The pre-request baseline (captured before the POST
    // fired) must not contain it, so it still counts as new.
    const baseline = new Set(useEventStore.getState().eventIds);
    act(() => {
      useEventStore.getState().addEvent(condensationEvent("cond-early"));
      useMetricsStore.getState().setMetrics({
        cost: null,
        max_budget_per_task: null,
        usage: {
          prompt_tokens: 0,
          completion_tokens: 0,
          cache_read_tokens: 0,
          cache_write_tokens: 0,
          context_window: 100_000,
          per_turn_token: 4_000,
        },
      });
    });

    const onComplete = vi.fn();
    renderHook(() =>
      useAwaitContextCompaction({
        beforeToken: 10_000,
        baselineEventIds: baseline,
        onComplete,
      }),
    );

    expect(onComplete).toHaveBeenCalledWith({
      beforeToken: 10_000,
      afterToken: 4_000,
      savedToken: 6_000,
      outcome: "compacted",
    });
  });
});
