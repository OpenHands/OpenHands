import { act, renderHook } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { useDragResize } from "#/hooks/use-drag-resize";

// Mock isMobileDevice to always return false so the hook uses the desktop
// event-listener path (document.addEventListener), which our test simulates.
vi.mock("#/utils/utils", async (importOriginal) => ({
  ...(await importOriginal<typeof import("#/utils/utils")>()),
  isMobileDevice: () => false,
}));

/**
 * Coverage for the top-grip resize direction fix (OHE-3062).
 *
 * The same chat-input component is used on the conversation page (where the
 * input is bottom-anchored) and the home page (where the input is centred).
 * On the conversation page, dragging the top grip makes the box grow upward —
 * correct. On the home page, the same drag would grow the box downward, which
 * is confusing, so the hook disables manual resizing entirely when the input
 * is not bottom-anchored.
 */
describe("useDragResize — disabled when not bottom-anchored", () => {
  let inputEl: HTMLDivElement;
  let wrapperEl: HTMLDivElement;
  let gripEl: HTMLDivElement;

  // Helper to simulate a drag from startY upward by `distance` px.
  const dragUp = (startY: number, distance: number) => {
    document.dispatchEvent(
      new MouseEvent("mousemove", { clientY: startY - distance }),
    );
    document.dispatchEvent(new MouseEvent("mouseup", { clientY: startY }));
  };

  beforeEach(() => {
    // Build the DOM structure that matches CustomChatInput:
    //   <div class="relative w-full" id="wrapper">   ← wrapper
    //     <div id="resize-grip">                      ← grip
    //     <div class="chat-input-container">           ← container
    //       <div id="inputEl">                         ← contentEditable (height target)
    wrapperEl = document.createElement("div");
    wrapperEl.id = "wrapper";

    gripEl = document.createElement("div");
    gripEl.id = "resize-grip";

    const containerEl = document.createElement("div");
    containerEl.className = "chat-input-container";

    inputEl = document.createElement("div");
    inputEl.style.height = "20px";

    containerEl.appendChild(inputEl);
    wrapperEl.appendChild(gripEl);
    wrapperEl.appendChild(containerEl);
    document.body.appendChild(wrapperEl);

    // jsdom doesn't implement offsetHeight; mock it to return the style height.
    Object.defineProperty(inputEl, "offsetHeight", {
      get() {
        return parseFloat(inputEl.style.height || "20");
      },
      configurable: true,
    });
  });

  afterEach(() => {
    wrapperEl.remove();
    vi.restoreAllMocks();
  });

  it("allows resizing when the input is bottom-anchored", () => {
    // Simulate a bottom-anchored layout: the wrapper's bottom is at the
    // viewport bottom.
    vi.spyOn(wrapperEl, "getBoundingClientRect").mockReturnValue({
      top: 668,
      bottom: 768, // == window.innerHeight
      left: 0,
      right: 0,
      width: 800,
      height: 100,
      x: 0,
      y: 668,
      toJSON: () => ({}),
    });
    vi.spyOn(window, "innerHeight", "get").mockReturnValue(768);

    const { result } = renderHook(() =>
      useDragResize({
        elementRef: { current: inputEl },
        minHeight: 20,
        maxHeight: 400,
      }),
    );

    act(() => {
      result.current.handleGripMouseDown(
        new MouseEvent("mousedown", {
          clientY: 100,
        }) as unknown as React.MouseEvent,
      );
    });

    // Drag up 30px → height should increase by 30
    act(() => dragUp(100, 30));

    expect(parseFloat(inputEl.style.height)).toBe(50);
  });

  it("does NOT resize when the input is not bottom-anchored (home page)", () => {
    // Simulate a centred layout: the wrapper's bottom is well above the
    // viewport bottom.
    vi.spyOn(wrapperEl, "getBoundingClientRect").mockReturnValue({
      top: 100,
      bottom: 200, // 200 << 768, not bottom-anchored
      left: 0,
      right: 0,
      width: 800,
      height: 100,
      x: 0,
      y: 100,
      toJSON: () => ({}),
    });
    vi.spyOn(window, "innerHeight", "get").mockReturnValue(768);

    const { result } = renderHook(() =>
      useDragResize({
        elementRef: { current: inputEl },
        minHeight: 20,
        maxHeight: 400,
      }),
    );

    act(() => {
      result.current.handleGripMouseDown(
        new MouseEvent("mousedown", {
          clientY: 100,
        }) as unknown as React.MouseEvent,
      );
    });

    // Attempt to drag up 30px — should be a no-op
    act(() => dragUp(100, 30));

    // Height should NOT have changed
    expect(parseFloat(inputEl.style.height)).toBe(20);
  });

  it("does NOT resize via touch when not bottom-anchored", () => {
    vi.spyOn(wrapperEl, "getBoundingClientRect").mockReturnValue({
      top: 100,
      bottom: 200,
      left: 0,
      right: 0,
      width: 800,
      height: 100,
      x: 0,
      y: 100,
      toJSON: () => ({}),
    });
    vi.spyOn(window, "innerHeight", "get").mockReturnValue(768);

    const { result } = renderHook(() =>
      useDragResize({
        elementRef: { current: inputEl },
        minHeight: 20,
        maxHeight: 400,
      }),
    );

    act(() => {
      result.current.handleGripTouchStart(
        new TouchEvent("touchstart", {
          touches: [{ clientY: 100 } as unknown as Touch],
        }) as unknown as React.TouchEvent,
      );
    });

    // Attempt to drag — should be a no-op
    act(() => dragUp(100, 30));

    expect(parseFloat(inputEl.style.height)).toBe(20);
  });
});
