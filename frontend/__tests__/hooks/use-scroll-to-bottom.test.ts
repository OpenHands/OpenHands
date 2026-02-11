import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useScrollToBottom } from "#/hooks/use-scroll-to-bottom";
import type { RefObject } from "react";

/**
 * Creates a mock scroll element with a trackable scrollTop setter.
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

  describe("no automatic scrolling on render", () => {
    it("does NOT scroll on initial render", () => {
      renderHook(() => useScrollToBottom(ref));

      // No useLayoutEffect means no automatic scroll-to-bottom
      expect(mock.scrollTopSetter).not.toHaveBeenCalled();
    });

    it("does NOT scroll when re-rendered (e.g., during resize)", () => {
      const { rerender } = renderHook(() => useScrollToBottom(ref));

      mock.state.scrollHeight = 1500;
      rerender();

      expect(mock.scrollTopSetter).not.toHaveBeenCalled();
    });
  });

  describe("scroll position tracking", () => {
    it("tracks hitBottom correctly via onChatBodyScroll", () => {
      const { result } = renderHook(() => useScrollToBottom(ref));

      // Position at bottom: scrollTop(480) + clientHeight(500) = 980 >= 1000 - 20
      mock.state.scrollTop = 480;
      act(() => {
        result.current.onChatBodyScroll(mock.element);
      });
      expect(result.current.hitBottom).toBe(true);

      // Position not at bottom: scrollTop(200) + clientHeight(500) = 700 < 980
      mock.state.scrollTop = 200;
      act(() => {
        result.current.onChatBodyScroll(mock.element);
      });
      expect(result.current.hitBottom).toBe(false);
    });

    it("disables autoScroll when user scrolls up", () => {
      const { result } = renderHook(() => useScrollToBottom(ref));

      // First scroll to establish prevScrollTopRef
      mock.state.scrollTop = 400;
      act(() => {
        result.current.onChatBodyScroll(mock.element);
      });

      // Scroll up (lower scrollTop than previous)
      mock.state.scrollTop = 200;
      act(() => {
        result.current.onChatBodyScroll(mock.element);
      });
      expect(result.current.autoScroll).toBe(false);
    });

    it("re-enables autoScroll when user reaches bottom", () => {
      const { result } = renderHook(() => useScrollToBottom(ref));

      // Scroll up to disable autoScroll
      mock.state.scrollTop = 400;
      act(() => {
        result.current.onChatBodyScroll(mock.element);
      });
      mock.state.scrollTop = 200;
      act(() => {
        result.current.onChatBodyScroll(mock.element);
      });
      expect(result.current.autoScroll).toBe(false);

      // Scroll to bottom
      mock.state.scrollTop = 500; // 500 + 500 = 1000 >= 980
      act(() => {
        result.current.onChatBodyScroll(mock.element);
      });
      expect(result.current.autoScroll).toBe(true);
    });
  });
});
