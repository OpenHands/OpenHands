import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { useStreamedText } from "#/hooks/use-streamed-text";

/** Drive `requestAnimationFrame` by hand so the reveal cadence is explicit. */
function manualFrames() {
  let now = 0;
  const callbacks = new Map<number, FrameRequestCallback>();
  let nextHandle = 1;

  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
    const handle = nextHandle;
    nextHandle += 1;
    callbacks.set(handle, callback);
    return handle;
  });
  vi.stubGlobal("cancelAnimationFrame", (handle: number) => {
    callbacks.delete(handle);
  });

  return {
    /** Advance one 16ms frame. */
    tick: (steps = 1) => {
      for (let i = 0; i < steps; i += 1) {
        now += 16;
        const scheduled = [...callbacks.values()];
        callbacks.clear();
        act(() => {
          scheduled.forEach((callback) => callback(now));
        });
      }
    },
    pending: () => callbacks.size,
  };
}

describe("useStreamedText", () => {
  let frames: ReturnType<typeof manualFrames>;

  beforeEach(() => {
    frames = manualFrames();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows the initial text immediately", () => {
    const { result } = renderHook(() => useStreamedText("already here"));
    expect(result.current).toBe("already here");
  });

  it("reveals appended text over frames instead of all at once", () => {
    const { result, rerender } = renderHook(
      ({ text }) => useStreamedText(text),
      { initialProps: { text: "Hi" } },
    );

    rerender({ text: `Hi${"x".repeat(200)}` });
    // Not committed on arrival — the clock has not ticked yet.
    expect(result.current).toBe("Hi");

    frames.tick();
    expect(result.current.length).toBeGreaterThan("Hi".length);
    expect(result.current.length).toBeLessThan(202);

    // It still catches up: the backlog drains within a short window.
    frames.tick(40);
    expect(result.current).toBe(`Hi${"x".repeat(200)}`);
    expect(frames.pending()).toBe(0);
  });

  it("snaps when the text is not an extension of what is shown", () => {
    const { result, rerender } = renderHook(
      ({ text }) => useStreamedText(text),
      { initialProps: { text: "first attempt" } },
    );

    // A higher attempt supersedes the slot; re-typing it would be wrong.
    rerender({ text: "a completely different retry" });
    expect(result.current).toBe("a completely different retry");
  });

  it("cancels the pending frame on unmount", () => {
    const { rerender, unmount } = renderHook(
      ({ text }) => useStreamedText(text),
      { initialProps: { text: "a" } },
    );
    rerender({ text: `a${"b".repeat(50)}` });
    expect(frames.pending()).toBe(1);

    unmount();
    expect(frames.pending()).toBe(0);
  });
});
