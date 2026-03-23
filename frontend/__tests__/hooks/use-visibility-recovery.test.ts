import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useVisibilityChange } from "#/hooks/use-visibility-change";

describe("useVisibilityChange", () => {
  beforeEach(() => {
    // Reset document.visibilityState to visible
    Object.defineProperty(document, "visibilityState", {
      value: "visible",
      writable: true,
      configurable: true,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("initial state", () => {
    it("should return isVisible=true when document is visible", () => {
      Object.defineProperty(document, "visibilityState", {
        value: "visible",
        writable: true,
      });

      const { result } = renderHook(() => useVisibilityChange());

      expect(result.current.isVisible).toBe(true);
    });

    it("should return isVisible=false when document is hidden", () => {
      Object.defineProperty(document, "visibilityState", {
        value: "hidden",
        writable: true,
      });

      const { result } = renderHook(() => useVisibilityChange());

      expect(result.current.isVisible).toBe(false);
    });
  });

  describe("visibility change events", () => {
    it("should update isVisible when visibility changes to hidden", () => {
      const { result } = renderHook(() => useVisibilityChange());

      expect(result.current.isVisible).toBe(true);

      // Simulate tab becoming hidden
      Object.defineProperty(document, "visibilityState", {
        value: "hidden",
        writable: true,
      });

      act(() => {
        document.dispatchEvent(new Event("visibilitychange"));
      });

      expect(result.current.isVisible).toBe(false);
    });

    it("should update isVisible when visibility changes to visible", () => {
      Object.defineProperty(document, "visibilityState", {
        value: "hidden",
        writable: true,
      });

      const { result } = renderHook(() => useVisibilityChange());

      expect(result.current.isVisible).toBe(false);

      // Simulate tab becoming visible
      Object.defineProperty(document, "visibilityState", {
        value: "visible",
        writable: true,
      });

      act(() => {
        document.dispatchEvent(new Event("visibilitychange"));
      });

      expect(result.current.isVisible).toBe(true);
    });
  });

  describe("callbacks", () => {
    it("should call onVisibilityChange with the new state", () => {
      const onVisibilityChange = vi.fn();

      renderHook(() => useVisibilityChange({ onVisibilityChange }));

      // Simulate tab becoming hidden
      Object.defineProperty(document, "visibilityState", {
        value: "hidden",
        writable: true,
      });

      act(() => {
        document.dispatchEvent(new Event("visibilitychange"));
      });

      expect(onVisibilityChange).toHaveBeenCalledWith("hidden");

      // Simulate tab becoming visible
      Object.defineProperty(document, "visibilityState", {
        value: "visible",
        writable: true,
      });

      act(() => {
        document.dispatchEvent(new Event("visibilitychange"));
      });

      expect(onVisibilityChange).toHaveBeenCalledWith("visible");
    });

    it("should call onVisible only when tab becomes visible", () => {
      const onVisible = vi.fn();
      const onHidden = vi.fn();

      renderHook(() => useVisibilityChange({ onVisible, onHidden }));

      // Simulate tab becoming hidden
      Object.defineProperty(document, "visibilityState", {
        value: "hidden",
        writable: true,
      });

      act(() => {
        document.dispatchEvent(new Event("visibilitychange"));
      });

      expect(onVisible).not.toHaveBeenCalled();
      expect(onHidden).toHaveBeenCalledTimes(1);

      // Simulate tab becoming visible
      Object.defineProperty(document, "visibilityState", {
        value: "visible",
        writable: true,
      });

      act(() => {
        document.dispatchEvent(new Event("visibilitychange"));
      });

      expect(onVisible).toHaveBeenCalledTimes(1);
      expect(onHidden).toHaveBeenCalledTimes(1);
    });

    it("should call onHidden only when tab becomes hidden", () => {
      const onHidden = vi.fn();

      renderHook(() => useVisibilityChange({ onHidden }));

      // Simulate tab becoming hidden
      Object.defineProperty(document, "visibilityState", {
        value: "hidden",
        writable: true,
      });

      act(() => {
        document.dispatchEvent(new Event("visibilitychange"));
      });

      expect(onHidden).toHaveBeenCalledTimes(1);

      // Simulate tab becoming visible (should not call onHidden)
      Object.defineProperty(document, "visibilityState", {
        value: "visible",
        writable: true,
      });

      act(() => {
        document.dispatchEvent(new Event("visibilitychange"));
      });

      expect(onHidden).toHaveBeenCalledTimes(1);
    });
  });

  describe("enabled option", () => {
    it("should not listen for events when enabled=false", () => {
      const onVisible = vi.fn();

      renderHook(() => useVisibilityChange({ onVisible, enabled: false }));

      // Simulate tab becoming visible
      Object.defineProperty(document, "visibilityState", {
        value: "visible",
        writable: true,
      });

      act(() => {
        document.dispatchEvent(new Event("visibilitychange"));
      });

      expect(onVisible).not.toHaveBeenCalled();
    });

    it("should start listening when enabled changes from false to true", () => {
      const onVisible = vi.fn();

      const { rerender } = renderHook(
        ({ enabled }) => useVisibilityChange({ onVisible, enabled }),
        { initialProps: { enabled: false } },
      );

      // Simulate event while disabled
      act(() => {
        document.dispatchEvent(new Event("visibilitychange"));
      });

      expect(onVisible).not.toHaveBeenCalled();

      // Enable the hook
      rerender({ enabled: true });

      // Now events should be captured
      act(() => {
        document.dispatchEvent(new Event("visibilitychange"));
      });

      expect(onVisible).toHaveBeenCalledTimes(1);
    });
  });

  describe("cleanup", () => {
    it("should remove event listener on unmount", () => {
      const addEventListenerSpy = vi.spyOn(document, "addEventListener");
      const removeEventListenerSpy = vi.spyOn(document, "removeEventListener");

      const { unmount } = renderHook(() => useVisibilityChange());

      expect(addEventListenerSpy).toHaveBeenCalledWith(
        "visibilitychange",
        expect.any(Function),
      );

      unmount();

      expect(removeEventListenerSpy).toHaveBeenCalledWith(
        "visibilitychange",
        expect.any(Function),
      );
    });

    it("should remove event listener when enabled changes to false", () => {
      const removeEventListenerSpy = vi.spyOn(document, "removeEventListener");

      const { rerender } = renderHook(
        ({ enabled }) => useVisibilityChange({ enabled }),
        { initialProps: { enabled: true } },
      );

      rerender({ enabled: false });

      expect(removeEventListenerSpy).toHaveBeenCalledWith(
        "visibilitychange",
        expect.any(Function),
      );
    });
  });

  describe("callback stability", () => {
    it("should handle callback updates without missing events", () => {
      const onVisible1 = vi.fn();
      const onVisible2 = vi.fn();

      const { rerender } = renderHook(
        ({ onVisible }) => useVisibilityChange({ onVisible }),
        { initialProps: { onVisible: onVisible1 } },
      );

      // Update callback
      rerender({ onVisible: onVisible2 });

      // Simulate visibility change
      act(() => {
        document.dispatchEvent(new Event("visibilitychange"));
      });

      expect(onVisible1).not.toHaveBeenCalled();
      expect(onVisible2).toHaveBeenCalledTimes(1);
    });
  });
});
