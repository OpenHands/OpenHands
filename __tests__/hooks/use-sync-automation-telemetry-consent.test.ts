import React from "react";
import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const syncTelemetryConsentMock = vi.fn();
const state = {
  backendId: "local-1",
  backendKind: "local" as "local" | "cloud",
  backendHost: "http://localhost:8000",
  backendApiKey: "key-1",
  backendConnectionRevision: 0,
  consent: "pending" as "pending" | "granted" | "denied",
  pendingRevocationId: null as string | null,
};
const listeners = new Set<() => void>();

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    syncTelemetryConsent: (...args: unknown[]) =>
      syncTelemetryConsentMock(...args),
  },
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: {
      id: state.backendId,
      kind: state.backendKind,
      host: state.backendHost,
      apiKey: state.backendApiKey,
      connectionRevision: state.backendConnectionRevision,
    },
  }),
}));

vi.mock("#/services/telemetry", () => ({
  getPendingLocalTelemetryRevocationId: () => state.pendingRevocationId,
  getTelemetryConsent: () => state.consent,
  subscribeTelemetryConsent: (listener: () => void) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
}));

// Import after mocks so the module sees the stubbed dependencies.
import { useSyncAutomationTelemetryConsent } from "#/hooks/use-sync-automation-telemetry-consent";

describe("useSyncAutomationTelemetryConsent", () => {
  beforeEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
    syncTelemetryConsentMock.mockResolvedValue(undefined);
    listeners.clear();
    state.backendId = "local-1";
    state.backendHost = "http://localhost:8000";
    state.backendApiKey = "key-1";
    state.backendConnectionRevision = 0;

    state.backendKind = "local";
    state.consent = "pending";
    state.pendingRevocationId = null;
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("does not call the local automation consent API while consent is pending", () => {
    renderHook(() => useSyncAutomationTelemetryConsent());

    expect(syncTelemetryConsentMock).not.toHaveBeenCalled();
  });

  it("revokes the pre-reset actor after a privacy clear", () => {
    state.pendingRevocationId = "user-before-reset";

    renderHook(() => useSyncAutomationTelemetryConsent());

    expect(syncTelemetryConsentMock).toHaveBeenCalledOnce();
    expect(syncTelemetryConsentMock).toHaveBeenCalledWith("denied");
  });

  it("does not call the local automation consent API for cloud backends", () => {
    state.backendKind = "cloud";
    state.consent = "granted";

    renderHook(() => useSyncAutomationTelemetryConsent());

    expect(syncTelemetryConsentMock).not.toHaveBeenCalled();
  });

  it("syncs resolved consent for a local backend once", () => {
    state.consent = "granted";
    const { rerender } = renderHook(() => useSyncAutomationTelemetryConsent());
    rerender();

    expect(syncTelemetryConsentMock).toHaveBeenCalledTimes(1);
    expect(syncTelemetryConsentMock).toHaveBeenCalledWith("granted");
  });

  it("syncs when consent resolves after mount", () => {
    renderHook(() => useSyncAutomationTelemetryConsent());

    act(() => {
      state.consent = "denied";
      listeners.forEach((listener) => listener());
    });

    expect(syncTelemetryConsentMock).toHaveBeenCalledTimes(1);
    expect(syncTelemetryConsentMock).toHaveBeenCalledWith("denied");
  });

  it("syncs the current consent when switching between local backends", () => {
    state.consent = "granted";
    const { rerender } = renderHook(() => useSyncAutomationTelemetryConsent());

    state.backendId = "local-2";
    rerender();

    expect(syncTelemetryConsentMock).toHaveBeenCalledTimes(2);
    expect(syncTelemetryConsentMock).toHaveBeenNthCalledWith(1, "granted");
    expect(syncTelemetryConsentMock).toHaveBeenNthCalledWith(2, "granted");
  });

  it("resyncs when the active local backend connection changes", () => {
    state.consent = "granted";
    const { rerender } = renderHook(() => useSyncAutomationTelemetryConsent());

    state.backendApiKey = "key-2";
    state.backendConnectionRevision = 1;
    rerender();

    expect(syncTelemetryConsentMock).toHaveBeenCalledTimes(2);
    expect(syncTelemetryConsentMock).toHaveBeenNthCalledWith(1, "granted");
    expect(syncTelemetryConsentMock).toHaveBeenNthCalledWith(2, "granted");
  });

  it("retries a failed sync with backoff until it succeeds", async () => {
    vi.useFakeTimers();
    state.consent = "granted";
    syncTelemetryConsentMock
      .mockRejectedValueOnce(new Error("temporarily unavailable"))
      .mockResolvedValueOnce(undefined);

    const { rerender } = renderHook(() => useSyncAutomationTelemetryConsent());
    await act(async () => Promise.resolve());

    expect(syncTelemetryConsentMock).toHaveBeenCalledTimes(1);

    await act(async () => vi.advanceTimersByTimeAsync(1000));
    expect(syncTelemetryConsentMock).toHaveBeenCalledTimes(2);

    rerender();
    expect(syncTelemetryConsentMock).toHaveBeenCalledTimes(2);
  });

  it("caps retry backoff while consent remains unacknowledged", async () => {
    vi.useFakeTimers();
    state.consent = "denied";
    syncTelemetryConsentMock.mockRejectedValue(new Error("offline"));

    renderHook(() => useSyncAutomationTelemetryConsent());
    await act(async () => Promise.resolve());

    const retryDelays = [1000, 2000, 4000, 8000, 16_000, 30_000, 30_000];
    for (const [index, delay] of retryDelays.entries()) {
      await act(async () => vi.advanceTimersByTimeAsync(delay - 1));
      expect(syncTelemetryConsentMock).toHaveBeenCalledTimes(index + 1);

      await act(async () => vi.advanceTimersByTimeAsync(1));
      expect(syncTelemetryConsentMock).toHaveBeenCalledTimes(index + 2);
    }
  });

  it("lets a consent change supersede an older in-flight sync", async () => {
    let rejectFirst: ((reason?: unknown) => void) | undefined;
    const firstRequest = new Promise<void>((_resolve, reject) => {
      rejectFirst = reject;
    });
    syncTelemetryConsentMock
      .mockReturnValueOnce(firstRequest)
      .mockResolvedValueOnce(undefined);
    state.consent = "granted";

    renderHook(() => useSyncAutomationTelemetryConsent());
    act(() => {
      state.consent = "denied";
      listeners.forEach((listener) => listener());
    });

    expect(syncTelemetryConsentMock).toHaveBeenCalledTimes(2);
    expect(syncTelemetryConsentMock).toHaveBeenNthCalledWith(1, "granted");
    expect(syncTelemetryConsentMock).toHaveBeenNthCalledWith(2, "denied");

    await act(async () => {
      rejectFirst?.(new Error("stale request failed"));
      await Promise.resolve();
    });

    expect(syncTelemetryConsentMock).toHaveBeenCalledTimes(2);
  });

  it("cancels the prior retry when the backend changes", async () => {
    vi.useFakeTimers();
    state.consent = "granted";
    syncTelemetryConsentMock
      .mockRejectedValueOnce(new Error("backend one unavailable"))
      .mockResolvedValueOnce(undefined);
    const { rerender } = renderHook(() => useSyncAutomationTelemetryConsent());
    await act(async () => Promise.resolve());

    state.backendId = "local-2";
    rerender();
    expect(syncTelemetryConsentMock).toHaveBeenCalledTimes(2);

    await act(async () => vi.advanceTimersByTimeAsync(30_000));
    expect(syncTelemetryConsentMock).toHaveBeenCalledTimes(2);
  });

  it("does not retry after unmount", async () => {
    vi.useFakeTimers();
    state.consent = "granted";
    syncTelemetryConsentMock.mockRejectedValueOnce(new Error("offline"));
    const { unmount } = renderHook(() => useSyncAutomationTelemetryConsent());
    await act(async () => Promise.resolve());

    unmount();
    await act(async () => vi.advanceTimersByTimeAsync(30_000));

    expect(syncTelemetryConsentMock).toHaveBeenCalledTimes(1);
  });

  it("does not duplicate an in-flight request during Strict Mode replay", () => {
    state.consent = "granted";
    syncTelemetryConsentMock.mockReturnValue(new Promise<void>(() => {}));
    const wrapper = ({ children }: React.PropsWithChildren) =>
      React.createElement(React.StrictMode, null, children);

    renderHook(() => useSyncAutomationTelemetryConsent(), { wrapper });

    expect(syncTelemetryConsentMock).toHaveBeenCalledTimes(1);
  });
});
