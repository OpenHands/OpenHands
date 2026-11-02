import type {
  MouseEvent as ReactMouseEvent,
  TouchEvent as ReactTouchEvent,
} from "react";
import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useDragResize } from "#/hooks/use-drag-resize";
import { isMobileDevice } from "#/utils/utils";

vi.mock("#/utils/utils", async (importOriginal) => ({
  ...(await importOriginal<typeof import("#/utils/utils")>()),
  isMobileDevice: vi.fn(),
}));

interface ResizeElementOptions {
  offsetHeight?: number;
  scrollHeight?: number;
}

const createResizeElement = ({
  offsetHeight = 200,
  scrollHeight = 100,
}: ResizeElementOptions = {}) => {
  const element = document.createElement("div");
  Object.defineProperties(element, {
    offsetHeight: { configurable: true, value: offsetHeight },
    scrollHeight: { configurable: true, value: scrollHeight },
  });
  document.body.appendChild(element);
  return element;
};

const createResizeGrip = () => {
  const grip = document.createElement("div");
  grip.id = "resize-grip";
  document.body.appendChild(grip);
  return grip;
};

const createTouchEvent = (type: string, clientYs: number[]) => {
  const event = new Event(type, { bubbles: true, cancelable: true });
  Object.defineProperty(event, "touches", {
    value: clientYs.map((clientY) => ({ clientY })),
  });
  return event as TouchEvent;
};

const beginMouseDrag = (
  handleGripMouseDown: (event: ReactMouseEvent) => void,
  clientY = 100,
) => {
  const preventDefault = vi.fn();
  act(() => {
    handleGripMouseDown({
      clientY,
      preventDefault,
    } as unknown as ReactMouseEvent);
  });
  expect(preventDefault).toHaveBeenCalledOnce();
};

const beginTouchDrag = (
  handleGripTouchStart: (event: ReactTouchEvent) => void,
  clientY = 100,
) => {
  const preventDefault = vi.fn();
  act(() => {
    handleGripTouchStart({
      touches: [{ clientY }],
      preventDefault,
    } as unknown as ReactTouchEvent);
  });
  expect(preventDefault).toHaveBeenCalledOnce();
};

const dispatchDocumentMouseMove = (clientY: number) => {
  const event = new MouseEvent("mousemove", {
    bubbles: true,
    cancelable: true,
    clientY,
  });
  act(() => document.dispatchEvent(event));
  return event;
};

const captureWindowErrors = (action: () => void) => {
  const errors: unknown[] = [];
  const handleError = (event: ErrorEvent) => {
    event.preventDefault();
    errors.push(event.error);
  };
  window.addEventListener("error", handleError);
  try {
    action();
  } catch (error) {
    errors.push(error);
  } finally {
    window.removeEventListener("error", handleError);
  }
  return errors;
};

beforeEach(() => {
  // The hook only enables manual resizing when the chat input is
  // "bottom-anchored" (its wrapper's bottom edge sits within
  // BOTTOM_ANCHORED_THRESHOLD px of the viewport bottom). jsdom reports a
  // zeroed getBoundingClientRect, so shrink the viewport to satisfy that
  // check and simulate the conversation-page layout where dragging is active.
  Object.defineProperty(window, "innerHeight", {
    configurable: true,
    value: 100,
  });
});

afterEach(() => {
  document.body.replaceChildren();
  vi.clearAllMocks();
});

describe("manual drag resizing", () => {
  it.each([648, 647])(
    "allows resizing at the bottom-anchor boundary but not just outside it: bottom=%s",
    (bottom) => {
      vi.mocked(isMobileDevice).mockReturnValue(false);
      Object.defineProperty(window, "innerHeight", {
        configurable: true,
        value: 768,
      });
      const wrapper = document.createElement("div");
      wrapper.getBoundingClientRect = () => ({ bottom }) as DOMRect;
      document.body.appendChild(wrapper);
      wrapper.appendChild(createResizeGrip());
      const element = createResizeElement();
      const onHeightChange = vi.fn();
      const { result } = renderHook(() =>
        useDragResize({
          elementRef: { current: element },
          minHeight: 100,
          maxHeight: 300,
          onHeightChange,
        }),
      );
      beginMouseDrag(result.current.handleGripMouseDown);
      dispatchDocumentMouseMove(80);
      act(() => document.dispatchEvent(new MouseEvent("mouseup")));
      if (bottom === 648) {
        expect(element.style.height).toBe("220px");
        expect(onHeightChange).toHaveBeenCalledExactlyOnceWith(220);
      } else {
        expect(element.style.height).toBe("");
        expect(onHeightChange).not.toHaveBeenCalled();
      }
    },
  );

  it.each([false, true])(
    "does not resize a centered home input on mobile=%s",
    (mobile) => {
      vi.mocked(isMobileDevice).mockReturnValue(mobile);
      Object.defineProperty(window, "innerHeight", {
        configurable: true,
        value: 768,
      });
      const wrapper = document.createElement("div");
      wrapper.getBoundingClientRect = () => ({ bottom: 200 }) as DOMRect;
      document.body.appendChild(wrapper);
      const grip = createResizeGrip();
      wrapper.appendChild(grip);
      const element = createResizeElement();
      const onHeightChange = vi.fn();
      const onGripDragStart = vi.fn();
      const { result } = renderHook(() =>
        useDragResize({
          elementRef: { current: element },
          minHeight: 100,
          maxHeight: 300,
          onHeightChange,
          onGripDragStart,
        }),
      );
      if (mobile) {
        beginTouchDrag(result.current.handleGripTouchStart);
        act(() => grip.dispatchEvent(createTouchEvent("touchmove", [80])));
        act(() => grip.dispatchEvent(createTouchEvent("touchend", [])));
      } else {
        beginMouseDrag(result.current.handleGripMouseDown);
        dispatchDocumentMouseMove(80);
        act(() => document.dispatchEvent(new MouseEvent("mouseup")));
      }
      expect(element.style.height).toBe("");
      expect(onHeightChange).not.toHaveBeenCalled();
      expect(onGripDragStart).not.toHaveBeenCalled();
    },
  );

  it("commits a desktop drag at the movement threshold and removes its listeners on release", () => {
    vi.mocked(isMobileDevice).mockReturnValue(false);
    const element = createResizeElement({
      offsetHeight: 200,
      scrollHeight: 220,
    });
    const onGripDragStart = vi.fn();
    const onGripDragEnd = vi.fn();
    const onHeightChange = vi.fn();
    const onReachedMinHeight = vi.fn();
    const { result } = renderHook(() =>
      useDragResize({
        elementRef: { current: element },
        minHeight: 100,
        maxHeight: 300,
        onGripDragStart,
        onGripDragEnd,
        onHeightChange,
        onReachedMinHeight,
      }),
    );

    beginMouseDrag(result.current.handleGripMouseDown);

    const subThresholdMove = dispatchDocumentMouseMove(99);
    expect(subThresholdMove.defaultPrevented).toBe(true);
    expect(onGripDragStart).not.toHaveBeenCalled();
    expect(onHeightChange).not.toHaveBeenCalled();
    expect(element.style.height).toBe("");

    dispatchDocumentMouseMove(98);
    expect(onGripDragStart).toHaveBeenCalledOnce();
    expect(onHeightChange).toHaveBeenLastCalledWith(202);
    expect(onReachedMinHeight).not.toHaveBeenCalled();
    expect(element.style.height).toBe("202px");
    expect(element.style.overflowY).toBe("auto");

    dispatchDocumentMouseMove(80);
    expect(onGripDragStart).toHaveBeenCalledOnce();
    expect(onHeightChange).toHaveBeenLastCalledWith(220);
    expect(element.style.height).toBe("220px");
    expect(element.style.overflowY).toBe("hidden");

    act(() => document.dispatchEvent(new MouseEvent("mouseup")));
    expect(onGripDragEnd).toHaveBeenCalledOnce();

    dispatchDocumentMouseMove(70);
    act(() => document.dispatchEvent(new MouseEvent("mouseup")));
    expect(onHeightChange).toHaveBeenCalledTimes(2);
    expect(onGripDragEnd).toHaveBeenCalledOnce();
  });

  it("clamps a desktop drag to the minimum and reports the minimum without optional height callbacks", () => {
    vi.mocked(isMobileDevice).mockReturnValue(false);
    const element = createResizeElement({
      offsetHeight: 200,
      scrollHeight: 150,
    });
    const onReachedMinHeight = vi.fn();
    const { result } = renderHook(() =>
      useDragResize({
        elementRef: { current: element },
        minHeight: 100,
        maxHeight: 300,
        onReachedMinHeight,
      }),
    );

    beginMouseDrag(result.current.handleGripMouseDown);
    dispatchDocumentMouseMove(400);

    expect(element.style.height).toBe("100px");
    expect(element.style.overflowY).toBe("auto");
    expect(onReachedMinHeight).toHaveBeenCalledOnce();

    const errors = captureWindowErrors(() => {
      act(() => document.dispatchEvent(new MouseEvent("mouseup")));
    });
    expect(errors).toEqual([]);
  });

  it("reports a height exactly within the minimum tolerance", () => {
    vi.mocked(isMobileDevice).mockReturnValue(false);
    const element = createResizeElement({
      offsetHeight: 200,
      scrollHeight: 0,
    });
    const onReachedMinHeight = vi.fn();
    const { result } = renderHook(() =>
      useDragResize({
        elementRef: { current: element },
        minHeight: 100,
        maxHeight: 300,
        onReachedMinHeight,
      }),
    );

    beginMouseDrag(result.current.handleGripMouseDown);
    dispatchDocumentMouseMove(198.5);

    expect(element.style.height).toBe("101.5px");
    expect(onReachedMinHeight).toHaveBeenCalledOnce();

    act(() => document.dispatchEvent(new MouseEvent("mouseup")));
  });

  it("clamps a desktop drag to the maximum and shows a scrollbar even when content fits", () => {
    vi.mocked(isMobileDevice).mockReturnValue(false);
    const element = createResizeElement({
      offsetHeight: 200,
      scrollHeight: 50,
    });
    const onHeightChange = vi.fn();
    const { result } = renderHook(() =>
      useDragResize({
        elementRef: { current: element },
        minHeight: 100,
        maxHeight: 300,
        onHeightChange,
      }),
    );

    beginMouseDrag(result.current.handleGripMouseDown);
    dispatchDocumentMouseMove(-200);

    expect(element.style.height).toBe("300px");
    expect(element.style.overflowY).toBe("auto");
    expect(onHeightChange).toHaveBeenCalledOnce();
    expect(onHeightChange).toHaveBeenCalledWith(300);

    act(() => document.dispatchEvent(new MouseEvent("mouseup")));
  });

  it("uses the minimum as the starting height when the element has no rendered height", () => {
    vi.mocked(isMobileDevice).mockReturnValue(false);
    const element = createResizeElement({ offsetHeight: 0, scrollHeight: 0 });
    const onHeightChange = vi.fn();
    const { result } = renderHook(() =>
      useDragResize({
        elementRef: { current: element },
        minHeight: 100,
        maxHeight: 300,
        onHeightChange,
      }),
    );

    beginMouseDrag(result.current.handleGripMouseDown);
    dispatchDocumentMouseMove(98);

    expect(element.style.height).toBe("102px");
    expect(element.style.overflowY).toBe("hidden");
    expect(onHeightChange).toHaveBeenCalledWith(102);

    act(() => document.dispatchEvent(new MouseEvent("mouseup")));
  });

  it("ends a committed desktop gesture safely if its resize element disappears", () => {
    vi.mocked(isMobileDevice).mockReturnValue(false);
    const elementRef: { current: HTMLElement | null } = { current: null };
    const onGripDragStart = vi.fn();
    const onGripDragEnd = vi.fn();
    const onHeightChange = vi.fn();
    const { result } = renderHook(() =>
      useDragResize({
        elementRef,
        minHeight: 100,
        maxHeight: 300,
        onGripDragStart,
        onGripDragEnd,
        onHeightChange,
      }),
    );

    beginMouseDrag(result.current.handleGripMouseDown);
    const errors = captureWindowErrors(() => dispatchDocumentMouseMove(98));
    act(() => document.dispatchEvent(new MouseEvent("mouseup")));

    expect(errors).toEqual([]);
    expect(onGripDragStart).toHaveBeenCalledOnce();
    expect(onGripDragEnd).toHaveBeenCalledOnce();
    expect(onHeightChange).not.toHaveBeenCalled();
  });

  it("does not report an uncommitted desktop press as a drag", () => {
    vi.mocked(isMobileDevice).mockReturnValue(false);
    const element = createResizeElement();
    const onGripDragStart = vi.fn();
    const onGripDragEnd = vi.fn();
    const { result } = renderHook(() =>
      useDragResize({
        elementRef: { current: element },
        minHeight: 100,
        maxHeight: 300,
        onGripDragStart,
        onGripDragEnd,
      }),
    );

    beginMouseDrag(result.current.handleGripMouseDown);
    act(() => document.dispatchEvent(new MouseEvent("mouseup")));

    expect(onGripDragStart).not.toHaveBeenCalled();
    expect(onGripDragEnd).not.toHaveBeenCalled();
  });

  it("uses the mouse coordinate when a mouse move has an empty touches collection", () => {
    vi.mocked(isMobileDevice).mockReturnValue(false);
    const element = createResizeElement();
    const onGripDragStart = vi.fn();
    const onHeightChange = vi.fn();
    const { result } = renderHook(() =>
      useDragResize({
        elementRef: { current: element },
        minHeight: 100,
        maxHeight: 300,
        onGripDragStart,
        onHeightChange,
      }),
    );

    beginMouseDrag(result.current.handleGripMouseDown);
    const moveEvent = new MouseEvent("mousemove", {
      bubbles: true,
      cancelable: true,
      clientY: 99,
    });
    Object.defineProperty(moveEvent, "touches", { value: [] });
    const errors = captureWindowErrors(() => {
      act(() => document.dispatchEvent(moveEvent));
    });

    expect(errors).toEqual([]);
    expect(moveEvent.defaultPrevented).toBe(true);
    expect(onGripDragStart).not.toHaveBeenCalled();
    expect(onHeightChange).not.toHaveBeenCalled();
    expect(element.style.height).toBe("");

    act(() => document.dispatchEvent(new MouseEvent("mouseup")));
  });

  it("does not start a mobile drag when the resize grip is absent", () => {
    vi.mocked(isMobileDevice).mockReturnValue(true);
    const element = createResizeElement();
    const onGripDragStart = vi.fn();
    const { result } = renderHook(() =>
      useDragResize({
        elementRef: { current: element },
        minHeight: 100,
        maxHeight: 300,
        onGripDragStart,
      }),
    );

    beginTouchDrag(result.current.handleGripTouchStart);
    act(() => document.dispatchEvent(createTouchEvent("touchmove", [80])));

    expect(onGripDragStart).not.toHaveBeenCalled();
    expect(element.style.height).toBe("");
  });

  it("resizes from touch input and reports the end of a mobile gesture", () => {
    vi.mocked(isMobileDevice).mockReturnValue(true);
    const element = createResizeElement({
      offsetHeight: 200,
      scrollHeight: 100,
    });
    const grip = createResizeGrip();
    const target = document.createElement("span");
    grip.appendChild(target);
    target.addEventListener("touchmove", (event) => event.stopPropagation());
    target.addEventListener("touchend", (event) => event.stopPropagation());
    const onGripDragStart = vi.fn();
    const onGripDragEnd = vi.fn();
    const onHeightChange = vi.fn();
    const { result } = renderHook(() =>
      useDragResize({
        elementRef: { current: element },
        minHeight: 100,
        maxHeight: 300,
        onGripDragStart,
        onGripDragEnd,
        onHeightChange,
      }),
    );

    beginTouchDrag(result.current.handleGripTouchStart);
    const moveEvent = createTouchEvent("touchmove", [80]);
    act(() => target.dispatchEvent(moveEvent));

    expect(moveEvent.defaultPrevented).toBe(true);
    expect(onGripDragStart).toHaveBeenCalledOnce();
    expect(onHeightChange).toHaveBeenCalledWith(220);
    expect(element.style.height).toBe("220px");
    expect(element.style.overflowY).toBe("hidden");

    act(() => target.dispatchEvent(createTouchEvent("touchend", [])));
    expect(onGripDragEnd).toHaveBeenCalledOnce();
    expect(onHeightChange).toHaveBeenCalledOnce();
  });

  it("stops handling mobile capture events after touchend", () => {
    vi.mocked(isMobileDevice).mockReturnValue(true);
    const element = createResizeElement();
    const grip = createResizeGrip();
    const onHeightChange = vi.fn();
    const onGripDragEnd = vi.fn();
    const { result } = renderHook(() =>
      useDragResize({
        elementRef: { current: element },
        minHeight: 100,
        maxHeight: 300,
        onHeightChange,
        onGripDragEnd,
      }),
    );

    beginTouchDrag(result.current.handleGripTouchStart);
    act(() => grip.dispatchEvent(createTouchEvent("touchmove", [80])));
    act(() => grip.dispatchEvent(createTouchEvent("touchend", [])));
    act(() => grip.dispatchEvent(createTouchEvent("touchmove", [70])));
    act(() => grip.dispatchEvent(createTouchEvent("touchend", [])));

    expect(onHeightChange).toHaveBeenCalledExactlyOnceWith(220);
    expect(onGripDragEnd).toHaveBeenCalledOnce();
    expect(element.style.height).toBe("220px");
  });

  it("finishes a mobile drag safely when the grip is detached before touchend", () => {
    vi.mocked(isMobileDevice).mockReturnValue(true);
    const element = createResizeElement();
    const grip = createResizeGrip();
    const onGripDragEnd = vi.fn();
    const { result } = renderHook(() =>
      useDragResize({
        elementRef: { current: element },
        minHeight: 100,
        maxHeight: 300,
        onGripDragEnd,
      }),
    );

    beginTouchDrag(result.current.handleGripTouchStart);
    act(() => grip.dispatchEvent(createTouchEvent("touchmove", [80])));
    grip.remove();
    createResizeGrip(); // A replacement must not receive the old grip's cleanup.
    const errors = captureWindowErrors(() => {
      act(() => grip.dispatchEvent(createTouchEvent("touchend", [])));
    });

    act(() => grip.dispatchEvent(createTouchEvent("touchmove", [70])));
    act(() => grip.dispatchEvent(createTouchEvent("touchend", [])));
    expect(errors).toEqual([]);
    expect(onGripDragEnd).toHaveBeenCalledOnce();
    expect(element.style.height).toBe("220px");
  });
});
