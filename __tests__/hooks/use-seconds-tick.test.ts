import { afterEach, describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { useSecondsTick } from "#/hooks/use-seconds-tick";

afterEach(() => {
  vi.useRealTimers();
});

describe("useSecondsTick", () => {
  it("does not create a timer when inactive", () => {
    vi.useFakeTimers();
    renderHook(() => useSecondsTick(false));
    expect(vi.getTimerCount()).toBe(0);
  });

  it("creates a timer when active", () => {
    vi.useFakeTimers();
    renderHook(() => useSecondsTick(true));
    expect(vi.getTimerCount()).toBeGreaterThan(0);
  });

  it("triggers a re-render after each second when active", async () => {
    vi.useFakeTimers();
    let renderCount = 0;
    renderHook(() => {
      renderCount += 1;
      useSecondsTick(true);
    });

    const initialRenders = renderCount;

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000);
    });
    expect(renderCount).toBeGreaterThan(initialRenders);

    const rendersAfterOneTick = renderCount;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000);
    });
    expect(renderCount).toBeGreaterThan(rendersAfterOneTick);
  });

  it("clears the timer when active transitions to inactive", async () => {
    vi.useFakeTimers();
    const { rerender } = renderHook(
      ({ active }: { active: boolean }) => useSecondsTick(active),
      { initialProps: { active: true } },
    );

    expect(vi.getTimerCount()).toBeGreaterThan(0);

    rerender({ active: false });
    expect(vi.getTimerCount()).toBe(0);
  });

  it("clears the timer on unmount", () => {
    vi.useFakeTimers();
    const { unmount } = renderHook(() => useSecondsTick(true));
    expect(vi.getTimerCount()).toBeGreaterThan(0);

    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });

  it("does not create a timer when initially inactive and remains inactive", async () => {
    vi.useFakeTimers();
    let renderCount = 0;
    renderHook(() => {
      renderCount += 1;
      useSecondsTick(false);
    });

    const initialRenders = renderCount;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000);
    });
    // No ticking — render count must not increase beyond React StrictMode
    // double-invocation (which is test-environment specific), so just check
    // that a timer did not cause additional renders.
    expect(vi.getTimerCount()).toBe(0);
    expect(renderCount).toBe(initialRenders);
  });
});
