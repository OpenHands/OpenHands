import { describe, expect, it, beforeEach, afterEach, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useBreakpoint } from "#/hooks/use-breakpoint";

// Helper to set window.innerWidth and dispatch resize event
function setWindowWidth(width: number) {
  Object.defineProperty(window, "innerWidth", {
    writable: true,
    configurable: true,
    value: width,
  });
  window.dispatchEvent(new Event("resize"));
}

describe("useBreakpoint", () => {
  const originalInnerWidth = window.innerWidth;

  beforeEach(() => {
    // Start at a known desktop width
    Object.defineProperty(window, "innerWidth", {
      writable: true,
      configurable: true,
      value: 1200,
    });
  });

  afterEach(() => {
    Object.defineProperty(window, "innerWidth", {
      writable: true,
      configurable: true,
      value: originalInnerWidth,
    });
  });

  it("returns false (not mobile) when window width is above the breakpoint", () => {
    Object.defineProperty(window, "innerWidth", { value: 1200 });
    const { result } = renderHook(() => useBreakpoint());
    expect(result.current).toBe(false);
  });

  it("returns true (mobile) when window width is at the breakpoint (1024)", () => {
    Object.defineProperty(window, "innerWidth", { value: 1024 });
    const { result } = renderHook(() => useBreakpoint());
    expect(result.current).toBe(true);
  });

  it("returns true (mobile) when window width is below the breakpoint", () => {
    Object.defineProperty(window, "innerWidth", { value: 800 });
    const { result } = renderHook(() => useBreakpoint());
    expect(result.current).toBe(true);
  });

  it("updates from false to true when window resizes below the breakpoint", () => {
    Object.defineProperty(window, "innerWidth", { value: 1200 });
    const { result } = renderHook(() => useBreakpoint());
    expect(result.current).toBe(false);

    act(() => {
      setWindowWidth(800);
    });

    expect(result.current).toBe(true);
  });

  it("updates from true to false when window resizes above the breakpoint", () => {
    Object.defineProperty(window, "innerWidth", { value: 800 });
    const { result } = renderHook(() => useBreakpoint());
    expect(result.current).toBe(true);

    act(() => {
      setWindowWidth(1200);
    });

    expect(result.current).toBe(false);
  });

  it("does NOT trigger re-render when width changes within the desktop range", () => {
    Object.defineProperty(window, "innerWidth", { value: 1200 });
    const renderCount = vi.fn();
    const { result } = renderHook(() => {
      renderCount();
      return useBreakpoint();
    });

    expect(result.current).toBe(false);
    const initialRenderCount = renderCount.mock.calls.length;

    // Resize within desktop range (still above 1024) — should NOT re-render
    act(() => {
      setWindowWidth(1300);
    });
    act(() => {
      setWindowWidth(1100);
    });
    act(() => {
      setWindowWidth(1025);
    });

    expect(result.current).toBe(false);
    // No additional renders beyond the initial render
    expect(renderCount.mock.calls.length).toBe(initialRenderCount);
  });

  it("does NOT trigger re-render when width changes within the mobile range", () => {
    Object.defineProperty(window, "innerWidth", { value: 800 });
    const renderCount = vi.fn();
    const { result } = renderHook(() => {
      renderCount();
      return useBreakpoint();
    });

    expect(result.current).toBe(true);
    const initialRenderCount = renderCount.mock.calls.length;

    // Resize within mobile range (still at or below 1024) — should NOT re-render
    act(() => {
      setWindowWidth(600);
    });
    act(() => {
      setWindowWidth(1024);
    });
    act(() => {
      setWindowWidth(900);
    });

    expect(result.current).toBe(true);
    expect(renderCount.mock.calls.length).toBe(initialRenderCount);
  });

  it("handles rapid resize across the breakpoint without issues", () => {
    Object.defineProperty(window, "innerWidth", { value: 1200 });
    const { result } = renderHook(() => useBreakpoint());
    expect(result.current).toBe(false);

    // Rapid toggles across the breakpoint
    act(() => {
      setWindowWidth(800);
    });
    expect(result.current).toBe(true);

    act(() => {
      setWindowWidth(1200);
    });
    expect(result.current).toBe(false);

    act(() => {
      setWindowWidth(1024);
    });
    expect(result.current).toBe(true);

    act(() => {
      setWindowWidth(1025);
    });
    expect(result.current).toBe(false);
  });

  it("cleans up the resize event listener on unmount", () => {
    const removeEventListenerSpy = vi.spyOn(window, "removeEventListener");
    const { unmount } = renderHook(() => useBreakpoint());

    unmount();

    expect(removeEventListenerSpy).toHaveBeenCalledWith(
      "resize",
      expect.any(Function),
    );
    removeEventListenerSpy.mockRestore();
  });

  it("accepts a custom breakpoint value", () => {
    Object.defineProperty(window, "innerWidth", { value: 768 });
    const { result } = renderHook(() => useBreakpoint(768));
    expect(result.current).toBe(true);

    act(() => {
      setWindowWidth(769);
    });
    expect(result.current).toBe(false);
  });
});
