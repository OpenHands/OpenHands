import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, renderHook } from "@testing-library/react";
import {
  useIsRevealingStreamedText,
  useStreamedText,
} from "#/hooks/use-streamed-text";

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
  // Arrivals are timed with `performance.now()`; keep it on the same clock.
  vi.spyOn(performance, "now").mockImplementation(() => now);

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
    vi.restoreAllMocks();
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

    // Under one character per frame at this pace, so give it a few.
    frames.tick(3);
    expect(result.current.length).toBeGreaterThan("Hi".length);
    expect(result.current.length).toBeLessThan(202);

    // A received burst must finish within 100ms plus one frame.
    frames.tick(4);
    expect(result.current).toBe(`Hi${"x".repeat(200)}`);
    expect(frames.pending()).toBe(0);
  });

  it("finishes each burst without accumulating a reading-pace backlog", () => {
    const { result, rerender } = renderHook(
      ({ text }) => useStreamedText(text),
      {
        initialProps: { text: "" },
      },
    );
    for (let i = 1; i <= 8; i += 1) {
      rerender({ text: "x".repeat(i * 12) });
      frames.tick(7);
      expect(result.current).toBe("x".repeat(i * 12));
    }
  });

  it("shows the final reply immediately and stops the reveal clock", () => {
    const { result, rerender } = renderHook(
      ({ text, streaming }) => useStreamedText(text, streaming),
      { initialProps: { text: "Hi", streaming: true } },
    );
    rerender({ text: "Hi" + "x".repeat(200), streaming: true });
    frames.tick();
    rerender({ text: "Hi" + "x".repeat(210), streaming: false });
    expect(result.current).toBe("Hi" + "x".repeat(210));
    expect(frames.pending()).toBe(0);
  });

  it("follows a fast model closely and catches up once it stops", () => {
    const { result, rerender } = renderHook(
      ({ text }) => useStreamedText(text),
      { initialProps: { text: "" } },
    );
    // A model streaming ~600 chars/s, one 10-char chunk per frame.
    let length = 0;
    const stream = (frameCount: number) => {
      for (let f = 0; f < frameCount; f += 1) {
        length += 10;
        rerender({ text: "x".repeat(length) });
        frames.tick();
      }
    };

    stream(300); // ~5s
    // Smoothed rather than shown as it lands, but never more than about half
    // a second of the model's output behind.
    expect(result.current.length).toBeLessThan(length);
    expect(length - result.current.length).toBeLessThan(600 * 0.6);

    frames.tick(7); // At most 112ms after the model stopped
    expect(result.current).toBe("x".repeat(length));
    expect(frames.pending()).toBe(0);
  });

  it("reports while text is still being revealed", () => {
    const revealing = renderHook(() => useIsRevealingStreamedText());
    const { rerender } = renderHook(({ text }) => useStreamedText(text), {
      initialProps: { text: "Hi" },
    });
    expect(revealing.result.current).toBe(false);

    rerender({ text: `Hi${"x".repeat(40)}` });
    expect(revealing.result.current).toBe(true);

    frames.tick(60);
    expect(revealing.result.current).toBe(false);
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
