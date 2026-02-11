import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useScrollToBottom } from "#/hooks/use-scroll-to-bottom";
import type { RefObject } from "react";

/**
 * Creates a mock scroll element with a trackable scrollTop setter.
 *
 * Why: We need to distinguish between "the hook read scrollTop" and
 * "the hook wrote scrollTop". The scrollTopSetter spy records every
 * write, letting us verify whether the useLayoutEffect actually
 * performed a scroll operation vs skipping it.
 *
 * state.scrollTop can be set directly (bypassing the spy) to position
 * the element for onChatBodyScroll calls without polluting the spy.
 */
function createMockScrollElement(initialScrollHeight = 1000) {
  const state = {
    scrollTop: 0,
    scrollHeight: initialScrollHeight,
    clientHeight: 500,
  };

  const scrollTopSetter = vi.fn((value: number) => {
    state.scrollTop = value;
  });

  const element = {
    get scrollTop() {
      return state.scrollTop;
    },
    set scrollTop(value: number) {
      scrollTopSetter(value);
    },
    get scrollHeight() {
      return state.scrollHeight;
    },
    get clientHeight() {
      return state.clientHeight;
    },
  } as unknown as HTMLDivElement;

  return { element, scrollTopSetter, state };
}

describe("useScrollToBottom", () => {
  let mock: ReturnType<typeof createMockScrollElement>;
  let ref: RefObject<HTMLDivElement>;

  beforeEach(() => {
    mock = createMockScrollElement(1000);
    ref = { current: mock.element } as RefObject<HTMLDivElement>;
  });

  describe("auto-scroll performance", () => {
    it("scrolls to bottom on initial render when autoscroll is true (default)", () => {
      renderHook(() => useScrollToBottom(ref));

      expect(mock.scrollTopSetter).toHaveBeenCalledWith(1000);
    });

    it("does NOT re-scroll when re-rendered with unchanged scrollHeight", () => {
      // Core performance test: during resize, scrollHeight doesn't change,
      // so the hook should skip the DOM scroll operation entirely.
      const { rerender } = renderHook(() => useScrollToBottom(ref));
      mock.scrollTopSetter.mockClear();

      // Re-render without changing scrollHeight (simulates resize re-render)
      rerender();

      expect(mock.scrollTopSetter).not.toHaveBeenCalled();
    });

    it("scrolls to bottom when scrollHeight increases (new content)", () => {
      const { rerender } = renderHook(() => useScrollToBottom(ref));
      mock.scrollTopSetter.mockClear();

      // Simulate new message arriving — content height grows
      mock.state.scrollHeight = 1500;
      rerender();

      expect(mock.scrollTopSetter).toHaveBeenCalledWith(1500);
    });

    it("does not scroll when autoscroll is disabled", () => {
      const { result, rerender } = renderHook(() => useScrollToBottom(ref));
      mock.scrollTopSetter.mockClear();

      // Disable autoscroll
      act(() => {
        result.current.setAutoScroll(false);
      });
      expect(result.current.autoScroll).toBe(false);
      mock.scrollTopSetter.mockClear();

      // New content arrives while autoscroll is off
      mock.state.scrollHeight = 1500;
      rerender();

      expect(mock.scrollTopSetter).not.toHaveBeenCalled();
    });

    it("scrolls after autoscroll is re-enabled when scrollHeight changed while disabled", () => {
      const { result, rerender } = renderHook(() => useScrollToBottom(ref));

      // Disable autoscroll
      act(() => {
        result.current.setAutoScroll(false);
      });
      expect(result.current.autoScroll).toBe(false);

      // New content arrives while autoscroll is off
      mock.state.scrollHeight = 2000;
      rerender();
      mock.scrollTopSetter.mockClear();

      // Re-enable autoscroll
      act(() => {
        result.current.setAutoScroll(true);
      });

      // The effect should scroll since scrollHeight changed while autoscroll was off
      expect(mock.scrollTopSetter).toHaveBeenCalledWith(2000);
    });
  });
});
