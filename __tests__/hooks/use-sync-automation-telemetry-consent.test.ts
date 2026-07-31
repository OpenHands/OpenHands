import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const syncTelemetryConsentMock = vi.fn();
const state = {
  backendId: "local-1",
  backendKind: "local" as "local" | "cloud",
  backendHost: "http://localhost:8000",
  backendApiKey: "key-1",
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
    vi.clearAllMocks();
    syncTelemetryConsentMock.mockResolvedValue(undefined);
    listeners.clear();
    state.backendId = "local-1";
    state.backendHost = "http://localhost:8000";
    state.backendApiKey = "key-1";

    state.backendKind = "local";
    state.consent = "pending";
    state.pendingRevocationId = null;
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
    rerender();

    expect(syncTelemetryConsentMock).toHaveBeenCalledTimes(2);
    expect(syncTelemetryConsentMock).toHaveBeenNthCalledWith(1, "granted");
    expect(syncTelemetryConsentMock).toHaveBeenNthCalledWith(2, "granted");
  });
});
