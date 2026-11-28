import { afterEach, describe, expect, it, vi, beforeEach } from "vitest";
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

  // A real element, so the hook can observe it and its children.
  const element = document.createElement("div");
  Object.defineProperties(element, {
    scrollTop: {
      get: () => state.scrollTop,
      set: scrollTopSetter,
    },
    scrollHeight: { get: () => state.scrollHeight },
    clientHeight: { get: () => state.clientHeight },
  });

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
  describe("following content that grows in place", () => {
    let frames: Map<number, FrameRequestCallback>;
    const tick = () => {
      const scheduled = [...frames.values()];
      frames.clear();
      act(() => scheduled.forEach((callback) => callback(0)));
    };

    beforeEach(() => {
      frames = new Map();
      let nextHandle = 1;
      vi.stubGlobal(
        "requestAnimationFrame",
        (callback: FrameRequestCallback) => {
          const handle = nextHandle;
          nextHandle += 1;
          frames.set(handle, callback);
          return handle;
        },
      );
      vi.stubGlobal("cancelAnimationFrame", (handle: number) => {
        frames.delete(handle);
      });
      // Pinned to the bottom: 1000 - 500.
      mock.state.scrollTop = 500;
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it("keeps the bottom in view as the content grows", () => {
      renderHook(() => useScrollToBottom(ref, true));

      mock.state.scrollHeight = 1100;
      tick();
      expect(mock.state.scrollTop).toBe(600);
    });

    it("stops following as soon as the user scrolls up", () => {
      const { result } = renderHook(() => useScrollToBottom(ref, true));
      tick();

      // The user scrolls up before React has re-rendered with autoScroll off.
      mock.state.scrollTop = 300;
      mock.state.scrollHeight = 1100;
      tick();
      expect(mock.state.scrollTop).toBe(300);
      expect(result.current.autoScroll).toBe(false);

      mock.state.scrollHeight = 1200;
      tick();
      expect(mock.state.scrollTop).toBe(300);
    });

    it("does not mistake the content shrinking for the user scrolling up", () => {
      const { result } = renderHook(() => useScrollToBottom(ref, true));
      tick();

      // The browser clamps scrollTop when the content gets shorter.
      mock.state.scrollHeight = 950;
      mock.state.scrollTop = 450;
      tick();
      mock.state.scrollHeight = 1050;
      tick();

      expect(result.current.autoScroll).toBe(true);
      expect(mock.state.scrollTop).toBe(550);
    });

    it("leaves the scroll position alone when the user is not pinned", () => {
      const { result } = renderHook(() => useScrollToBottom(ref, true));
      act(() => result.current.setAutoScroll(false));
      mock.state.scrollTop = 100;

      mock.state.scrollHeight = 1100;
      tick();
      expect(mock.state.scrollTop).toBe(100);
    });

    it("only runs while content is growing", () => {
      const { rerender } = renderHook(
        ({ follow }) => useScrollToBottom(ref, follow),
        { initialProps: { follow: false } },
      );
      expect(frames.size).toBe(0);

      rerender({ follow: true });
      expect(frames.size).toBe(1);

      rerender({ follow: false });
      expect(frames.size).toBe(0);
    });
  });

  describe("staying at the bottom when the transcript resizes", () => {
    let observed: Set<Element>;
    let notifyResize: () => void;

    beforeEach(() => {
      observed = new Set();
      vi.stubGlobal(
        "ResizeObserver",
        class {
          constructor(callback: () => void) {
            notifyResize = () => act(() => callback());
          }

          observe = (target: Element) => observed.add(target);

          unobserve = (target: Element) => observed.delete(target);

          disconnect = () => observed.clear();
        },
      );
      // The issue's numbers: at the bottom at 1440x1000, then 390x844.
      mock.state.scrollHeight = 1888;
      mock.state.clientHeight = 814;
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    const narrow = () => {
      mock.state.scrollHeight = 1912;
      mock.state.clientHeight = 650;
      notifyResize();
    };

    it("re-pins a reader who was at the bottom when the container reflows taller", () => {
      const { result } = renderHook(() => useScrollToBottom(ref));
      mock.state.scrollTop = 1074;
      act(() => result.current.onChatBodyScroll(mock.element));

      narrow();

      expect(mock.state.scrollTop).toBe(1262);
    });

    it("keeps a reader who scrolled up in place", () => {
      const { result } = renderHook(() => useScrollToBottom(ref));
      mock.state.scrollTop = 1074;
      act(() => result.current.onChatBodyScroll(mock.element));
      mock.state.scrollTop = 600;
      act(() => result.current.onChatBodyScroll(mock.element));

      narrow();

      expect(mock.state.scrollTop).toBe(600);
    });

    it("stays pinned when a scroll event reads the gap after the content grew", () => {
      const { result } = renderHook(() => useScrollToBottom(ref));
      mock.state.scrollTop = 1074;
      act(() => result.current.onChatBodyScroll(mock.element));

      // The scroll lands on the old bottom, but its event is dispatched after
      // the content has grown again, so the element reads a gap.
      mock.state.scrollTop = 1080;
      mock.state.scrollHeight = 1924;
      act(() => result.current.onChatBodyScroll(mock.element));
      notifyResize();

      expect(mock.state.scrollTop).toBe(1110);
    });

    it("observes the container and the rows inside its content wrappers", async () => {
      // Rows render inside `display: contents` wrappers, which have no box.
      // Styles only compute for a connected element.
      document.body.appendChild(mock.element);
      const wrapper = mock.element.appendChild(document.createElement("div"));
      wrapper.style.display = "contents";
      const first = wrapper.appendChild(document.createElement("article"));
      const { unmount } = renderHook(() => useScrollToBottom(ref));
      expect(observed).toEqual(new Set([mock.element, first]));

      const later = wrapper.appendChild(document.createElement("article"));
      first.remove();
      await vi.waitFor(() =>
        expect(observed).toEqual(new Set([mock.element, later])),
      );

      unmount();
      expect(observed.size).toBe(0);
      mock.element.remove();
    });
  });
});
