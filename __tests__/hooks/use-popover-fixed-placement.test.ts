import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { usePopoverFixedPlacement } from "#/hooks/use-popover-fixed-placement";

const originalInnerWidth = Object.getOwnPropertyDescriptor(
  window,
  "innerWidth",
);

function setInnerWidth(value: number) {
  Object.defineProperty(window, "innerWidth", {
    configurable: true,
    value,
  });
}

function createTrigger({ bottom, right }: { bottom: number; right: number }) {
  let rect = { bottom, right };
  const element = document.createElement("button");
  vi.spyOn(element, "getBoundingClientRect").mockImplementation(
    () => rect as DOMRect,
  );

  return {
    element,
    setRect(nextRect: { bottom: number; right: number }) {
      rect = nextRect;
    },
  };
}

afterEach(() => {
  vi.restoreAllMocks();
  if (originalInnerWidth) {
    Object.defineProperty(window, "innerWidth", originalInnerWidth);
  }
});

describe("usePopoverFixedPlacement", () => {
  it("measures below the trigger with the default width", () => {
    setInnerWidth(1000);
    const trigger = createTrigger({ bottom: 100, right: 500 });
    const triggerRef = { current: trigger.element as HTMLElement | null };

    const { result } = renderHook(() =>
      usePopoverFixedPlacement(triggerRef, { enabled: true, open: true }),
    );

    expect(result.current).toEqual({
      top: 104,
      left: 244,
      width: 256,
    });
  });

  it("clamps the popover to the left and right viewport gutters", () => {
    setInnerWidth(1000);
    const trigger = createTrigger({ bottom: 20, right: 200 });
    const triggerRef = { current: trigger.element as HTMLElement | null };
    const { result } = renderHook(() =>
      usePopoverFixedPlacement(triggerRef, {
        enabled: true,
        open: true,
        targetWidth: 256,
      }),
    );

    expect(result.current).toEqual({ top: 24, left: 8, width: 256 });

    trigger.setRect({ bottom: 30, right: 1100 });
    act(() => window.dispatchEvent(new Event("scroll")));
    expect(result.current).toEqual({ top: 34, left: 736, width: 256 });

    trigger.setRect({ bottom: 25, right: 1000 });
    act(() => window.dispatchEvent(new Event("scroll")));
    expect(result.current).toEqual({ top: 29, left: 736, width: 256 });

    setInnerWidth(200);
    act(() => window.dispatchEvent(new Event("resize")));
    expect(result.current).toEqual({ top: 29, left: 8, width: 184 });
  });

  it("remeasures when the target width changes", () => {
    setInnerWidth(1000);
    const trigger = createTrigger({ bottom: 40, right: 500 });
    const triggerRef = { current: trigger.element as HTMLElement | null };
    const { result, rerender } = renderHook(
      ({ targetWidth }: { targetWidth: number }) =>
        usePopoverFixedPlacement(triggerRef, {
          enabled: true,
          open: true,
          targetWidth,
        }),
      { initialProps: { targetWidth: 256 } },
    );

    expect(result.current).toEqual({ top: 44, left: 244, width: 256 });

    rerender({ targetWidth: 100 });
    expect(result.current).toEqual({ top: 44, left: 400, width: 100 });
  });

  it("waits for a trigger and measures it on the next captured scroll", () => {
    setInnerWidth(1000);
    const trigger = createTrigger({ bottom: 60, right: 600 });
    const triggerRef = { current: null as HTMLElement | null };
    const { result } = renderHook(() =>
      usePopoverFixedPlacement(triggerRef, {
        enabled: true,
        open: true,
        targetWidth: 200,
      }),
    );

    expect(result.current).toBeNull();

    triggerRef.current = trigger.element;
    act(() => window.dispatchEvent(new Event("scroll")));
    expect(result.current).toEqual({ top: 64, left: 400, width: 200 });
  });

  it("clears its box whenever fixed placement is closed or disabled", () => {
    setInnerWidth(1000);
    const trigger = createTrigger({ bottom: 80, right: 400 });
    const triggerRef = { current: trigger.element as HTMLElement | null };
    const { result, rerender } = renderHook(
      ({ enabled, open }: { enabled: boolean; open: boolean }) =>
        usePopoverFixedPlacement(triggerRef, { enabled, open }),
      { initialProps: { enabled: true, open: true } },
    );

    expect(result.current).not.toBeNull();

    rerender({ enabled: true, open: false });
    expect(result.current).toBeNull();

    rerender({ enabled: true, open: true });
    expect(result.current).not.toBeNull();

    rerender({ enabled: false, open: true });
    expect(result.current).toBeNull();
  });

  it("registers resize and capture-scroll listeners and removes them", () => {
    setInnerWidth(1000);
    const addEventListener = vi.spyOn(window, "addEventListener");
    const removeEventListener = vi.spyOn(window, "removeEventListener");
    const trigger = createTrigger({ bottom: 10, right: 300 });
    const triggerRef = { current: trigger.element as HTMLElement | null };
    const { unmount } = renderHook(() =>
      usePopoverFixedPlacement(triggerRef, { enabled: true, open: true }),
    );

    expect(addEventListener).toHaveBeenCalledWith(
      "resize",
      expect.any(Function),
    );
    expect(addEventListener).toHaveBeenCalledWith(
      "scroll",
      expect.any(Function),
      true,
    );

    const resizeListener = addEventListener.mock.calls.find(
      ([type]) => type === "resize",
    )?.[1];
    const scrollListener = addEventListener.mock.calls.find(
      ([type]) => type === "scroll",
    )?.[1];

    unmount();

    expect(removeEventListener).toHaveBeenCalledWith("resize", resizeListener);
    expect(removeEventListener).toHaveBeenCalledWith(
      "scroll",
      scrollListener,
      true,
    );
  });
});
