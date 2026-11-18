import { describe, expect, it, beforeEach, afterEach, vi } from "vitest";
import {
  getPinnedHomeRouteKey,
  PINNED_HOME_ROUTE_KEY,
} from "#/hooks/use-pinned-home-route";
import { act, renderHook } from "@testing-library/react";
import { type MouseEvent as ReactMouseEvent, type RefObject } from "react";
import { useResizablePanels } from "#/hooks/use-resizable-panels";
import { useScrollToBottom } from "#/hooks/use-scroll-to-bottom";
import {
  useIsRevealingStreamedText,
  useStreamedText,
} from "#/hooks/use-streamed-text";
import { createInstance } from "i18next";
import { getI18n, setI18n } from "#/i18n";
import { useSyncDocumentLanguage } from "#/hooks/use-sync-document-language";

describe("getPinnedHomeRouteKey", () => {
  it("scopes the storage key by backend and org", () => {
    expect(getPinnedHomeRouteKey("backend-a", "org-1")).toBe(
      `${PINNED_HOME_ROUTE_KEY}:backend-a:org-1`,
    );
    expect(getPinnedHomeRouteKey("backend-a", null)).toBe(
      `${PINNED_HOME_ROUTE_KEY}:backend-a:-`,
    );
    expect(getPinnedHomeRouteKey("backend-a", "org-1")).not.toBe(
      getPinnedHomeRouteKey("backend-b", "org-1"),
    );
  });
});

describe("useResizablePanels", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("clamps a stale persisted width that is below minLeftWidth on read", () => {
    window.localStorage.setItem("test-panel-width", JSON.stringify(10));

    const { result } = renderHook(() =>
      useResizablePanels({
        defaultLeftWidth: 50,
        minLeftWidth: 30,
        maxLeftWidth: 80,
        storageKey: "test-panel-width",
      }),
    );

    expect(result.current.leftWidth).toBe(30);
    expect(result.current.rightWidth).toBe(70);
  });

  it("clamps a stale persisted width that is above maxLeftWidth on read", () => {
    window.localStorage.setItem("test-panel-width", JSON.stringify(95));

    const { result } = renderHook(() =>
      useResizablePanels({
        defaultLeftWidth: 50,
        minLeftWidth: 30,
        maxLeftWidth: 80,
        storageKey: "test-panel-width",
      }),
    );

    expect(result.current.leftWidth).toBe(80);
    expect(result.current.rightWidth).toBe(20);
  });

  it("uses the persisted value as-is when within range", () => {
    window.localStorage.setItem("test-panel-width", JSON.stringify(60));

    const { result } = renderHook(() =>
      useResizablePanels({
        defaultLeftWidth: 50,
        minLeftWidth: 30,
        maxLeftWidth: 80,
        storageKey: "test-panel-width",
      }),
    );

    expect(result.current.leftWidth).toBe(60);
    expect(result.current.rightWidth).toBe(40);
  });

  it("mounts a full-viewport drag shield while dragging so iframe-heavy panes do not steal pointer events", () => {
    const { result } = renderHook(() =>
      useResizablePanels({
        defaultLeftWidth: 50,
        minLeftWidth: 30,
        maxLeftWidth: 80,
        storageKey: "test-panel-width",
      }),
    );

    expect(document.querySelector("[data-panel-drag-shield]")).toBeNull();

    act(() => {
      result.current.handleMouseDown({
        preventDefault: () => {},
      } as unknown as ReactMouseEvent<HTMLDivElement>);
    });

    const shield = document.querySelector("[data-panel-drag-shield]");
    expect(shield).not.toBeNull();
    expect(shield).toBeInstanceOf(HTMLDivElement);

    act(() => {
      document.dispatchEvent(new MouseEvent("mouseup"));
    });

    expect(document.querySelector("[data-panel-drag-shield]")).toBeNull();
  });
});

describe("use-scroll-to-bottom", () => {
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
  });
});

describe("use-streamed-text", () => {
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
});

// Screen readers pick their voice from <html lang>; right-to-left scripts
// also need <html dir> (#17909).
describe("useSyncDocumentLanguage", () => {
  beforeEach(async () => {
    const instance = createInstance();
    await instance.init({ lng: "de", resources: {} });
    setI18n(instance);
    document.documentElement.lang = "en";
    document.documentElement.removeAttribute("dir");
  });

  afterEach(() => {
    setI18n(null);
    document.documentElement.lang = "en";
    document.documentElement.removeAttribute("dir");
  });

  it("applies the active language on mount", () => {
    renderHook(() => useSyncDocumentLanguage());

    expect(document.documentElement.lang).toBe("de");
    expect(document.documentElement.dir).toBe("ltr");
  });

  it("follows language changes, setting rtl for Arabic", async () => {
    renderHook(() => useSyncDocumentLanguage());

    await act(() => getI18n().changeLanguage("ar"));
    expect(document.documentElement.lang).toBe("ar");
    expect(document.documentElement.dir).toBe("rtl");

    await act(() => getI18n().changeLanguage("fr"));
    expect(document.documentElement.lang).toBe("fr");
    expect(document.documentElement.dir).toBe("ltr");
  });

  it("stops following language changes after unmount", async () => {
    const { unmount } = renderHook(() => useSyncDocumentLanguage());
    unmount();

    await act(() => getI18n().changeLanguage("ar"));

    expect(document.documentElement.lang).toBe("de");
    expect(document.documentElement.dir).toBe("ltr");
  });
});
