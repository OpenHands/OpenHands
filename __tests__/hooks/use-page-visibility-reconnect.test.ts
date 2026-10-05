import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { usePageVisibilityReconnect } from "#/hooks/use-page-visibility-reconnect";

const MOBILE_USER_AGENT =
  "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15";
const DESKTOP_USER_AGENT =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36";

function setUserAgent(userAgent: string) {
  vi.spyOn(window.navigator, "userAgent", "get").mockReturnValue(userAgent);
}

function setVisibility(state: "visible" | "hidden") {
  Object.defineProperty(document, "visibilityState", {
    value: state,
    configurable: true,
  });
  document.dispatchEvent(new Event("visibilitychange"));
}

describe("usePageVisibilityReconnect", () => {
  const originalVisibilityDescriptor = Object.getOwnPropertyDescriptor(
    Document.prototype,
    "visibilityState",
  );

  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    if (originalVisibilityDescriptor) {
      Object.defineProperty(
        document,
        "visibilityState",
        originalVisibilityDescriptor,
      );
    }
  });

  it("disconnects on a mobile user agent only after the hide debounce elapses", () => {
    setUserAgent(MOBILE_USER_AGENT);
    const disconnect = vi.fn();
    const reconnectIfStale = vi.fn();
    renderHook(() =>
      usePageVisibilityReconnect({
        enabled: true,
        disconnect,
        reconnectIfStale,
      }),
    );

    act(() => setVisibility("hidden"));
    expect(disconnect).not.toHaveBeenCalled();

    act(() => vi.advanceTimersByTime(2_999));
    expect(disconnect).not.toHaveBeenCalled();

    act(() => vi.advanceTimersByTime(1));
    expect(disconnect).toHaveBeenCalledOnce();
  });

  it("cancels the pending disconnect when the tab returns before the debounce fires", () => {
    setUserAgent(MOBILE_USER_AGENT);
    const disconnect = vi.fn();
    const reconnectIfStale = vi.fn();
    renderHook(() =>
      usePageVisibilityReconnect({
        enabled: true,
        disconnect,
        reconnectIfStale,
      }),
    );

    act(() => {
      setVisibility("hidden");
      vi.advanceTimersByTime(1_000);
      setVisibility("visible");
      vi.advanceTimersByTime(10_000);
    });

    expect(disconnect).not.toHaveBeenCalled();
    // The visible transition still runs the health check.
    expect(reconnectIfStale).toHaveBeenCalled();
  });

  it("never disconnects on a desktop user agent, however long the tab stays hidden", () => {
    setUserAgent(DESKTOP_USER_AGENT);
    const disconnect = vi.fn();
    const reconnectIfStale = vi.fn();
    renderHook(() =>
      usePageVisibilityReconnect({
        enabled: true,
        disconnect,
        reconnectIfStale,
      }),
    );

    act(() => {
      setVisibility("hidden");
      vi.advanceTimersByTime(60_000);
    });

    expect(disconnect).not.toHaveBeenCalled();
  });

  it("restarts the debounce on repeated flapping instead of stacking timers", () => {
    setUserAgent(MOBILE_USER_AGENT);
    const disconnect = vi.fn();
    const reconnectIfStale = vi.fn();
    renderHook(() =>
      usePageVisibilityReconnect({
        enabled: true,
        disconnect,
        reconnectIfStale,
      }),
    );

    act(() => {
      setVisibility("hidden");
      vi.advanceTimersByTime(2_000);
      setVisibility("visible");
      setVisibility("hidden");
      vi.advanceTimersByTime(2_000);
    });
    // 4s of cumulative "hidden" time has elapsed, but each hide restarted a
    // fresh 3s window, so still nothing torn down.
    expect(disconnect).not.toHaveBeenCalled();

    act(() => vi.advanceTimersByTime(1_000));
    expect(disconnect).toHaveBeenCalledOnce();
  });

  it("runs the foreground health check on every visible transition, even without a prior disconnect", () => {
    setUserAgent(DESKTOP_USER_AGENT);
    const disconnect = vi.fn();
    const reconnectIfStale = vi.fn();
    renderHook(() =>
      usePageVisibilityReconnect({
        enabled: true,
        disconnect,
        reconnectIfStale,
      }),
    );

    act(() => {
      setVisibility("hidden");
      setVisibility("visible");
    });

    expect(reconnectIfStale).toHaveBeenCalledOnce();
  });

  it("reacts to pagehide/pageshow in addition to visibilitychange", () => {
    setUserAgent(MOBILE_USER_AGENT);
    const disconnect = vi.fn();
    const reconnectIfStale = vi.fn();
    renderHook(() =>
      usePageVisibilityReconnect({
        enabled: true,
        disconnect,
        reconnectIfStale,
      }),
    );

    act(() => {
      window.dispatchEvent(new Event("pagehide"));
      vi.advanceTimersByTime(3_000);
    });
    expect(disconnect).toHaveBeenCalledOnce();

    act(() => window.dispatchEvent(new Event("pageshow")));
    expect(reconnectIfStale).toHaveBeenCalledOnce();
  });

  it("disconnects immediately on `freeze`, bypassing the hide debounce, even on desktop", () => {
    setUserAgent(DESKTOP_USER_AGENT);
    const disconnect = vi.fn();
    const reconnectIfStale = vi.fn();
    renderHook(() =>
      usePageVisibilityReconnect({
        enabled: true,
        disconnect,
        reconnectIfStale,
      }),
    );

    act(() => document.dispatchEvent(new Event("freeze")));

    expect(disconnect).toHaveBeenCalledOnce();
  });

  it("cancels a pending debounced disconnect once `freeze` has already handled it", () => {
    setUserAgent(MOBILE_USER_AGENT);
    const disconnect = vi.fn();
    const reconnectIfStale = vi.fn();
    renderHook(() =>
      usePageVisibilityReconnect({
        enabled: true,
        disconnect,
        reconnectIfStale,
      }),
    );

    act(() => {
      setVisibility("hidden");
      document.dispatchEvent(new Event("freeze"));
      vi.advanceTimersByTime(10_000);
    });

    expect(disconnect).toHaveBeenCalledOnce();
  });

  it("reconnects immediately on `resume`", () => {
    setUserAgent(MOBILE_USER_AGENT);
    const disconnect = vi.fn();
    const reconnectIfStale = vi.fn();
    renderHook(() =>
      usePageVisibilityReconnect({
        enabled: true,
        disconnect,
        reconnectIfStale,
      }),
    );

    act(() => document.dispatchEvent(new Event("resume")));

    expect(reconnectIfStale).toHaveBeenCalledOnce();
  });

  it("does nothing when disabled", () => {
    setUserAgent(MOBILE_USER_AGENT);
    const disconnect = vi.fn();
    const reconnectIfStale = vi.fn();
    renderHook(() =>
      usePageVisibilityReconnect({
        enabled: false,
        disconnect,
        reconnectIfStale,
      }),
    );

    act(() => {
      setVisibility("hidden");
      vi.advanceTimersByTime(10_000);
      setVisibility("visible");
    });

    expect(disconnect).not.toHaveBeenCalled();
    expect(reconnectIfStale).not.toHaveBeenCalled();
  });

  it("stops reacting once unmounted", () => {
    setUserAgent(MOBILE_USER_AGENT);
    const disconnect = vi.fn();
    const reconnectIfStale = vi.fn();
    const { unmount } = renderHook(() =>
      usePageVisibilityReconnect({
        enabled: true,
        disconnect,
        reconnectIfStale,
      }),
    );

    unmount();
    act(() => {
      setVisibility("hidden");
      vi.advanceTimersByTime(10_000);
    });

    expect(disconnect).not.toHaveBeenCalled();
  });
});
