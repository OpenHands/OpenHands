import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const useSettingsMock = vi.fn();
const saveSettingsMock = vi.fn();
const useSaveSettingsMock = vi.fn();
const clearPendingCloudTelemetryConsentMock = vi.fn();
const setTelemetryConsentMock = vi.fn();
const state = {
  backendId: "backend-1",
  backendKind: "cloud" as "cloud" | "local",
  pendingConsent: null as "granted" | "denied" | null,
  isSavingSettings: false,
};
let pendingConsentListener: (() => void) | null = null;

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => useSettingsMock(),
}));

vi.mock("#/hooks/mutation/use-save-settings", () => ({
  useSaveSettings: (...args: unknown[]) => {
    useSaveSettingsMock(...args);
    return {
      mutate: saveSettingsMock,
      isPending: state.isSavingSettings,
    };
  },
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: state.backendId, kind: state.backendKind },
  }),
}));

vi.mock("#/services/telemetry", () => ({
  clearPendingCloudTelemetryConsent: (...args: unknown[]) =>
    clearPendingCloudTelemetryConsentMock(...args),
  getPendingCloudTelemetryConsent: () => state.pendingConsent,
  setTelemetryConsent: (...args: unknown[]) => setTelemetryConsentMock(...args),
  subscribeTelemetryConsent: (listener: () => void) => {
    pendingConsentListener = listener;
    return () => {
      pendingConsentListener = null;
    };
  },
}));

// Import after mocks so the module sees the stubbed dependencies.
import { useSyncTelemetryConsent } from "#/hooks/use-sync-telemetry-consent";

describe("useSyncTelemetryConsent", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    state.backendId = "backend-1";
    state.backendKind = "cloud";
    state.pendingConsent = null;
    state.isSavingSettings = false;
    pendingConsentListener = null;
  });

  it("calls opt-out when user_consents_to_analytics is null", () => {
    useSettingsMock.mockReturnValue({
      data: { user_consents_to_analytics: null },
    });

    renderHook(() => useSyncTelemetryConsent());

    expect(setTelemetryConsentMock).toHaveBeenCalledWith("denied", {
      syncToCloud: false,
    });
  });

  it("calls opt-out when user_consents_to_analytics is false", () => {
    useSettingsMock.mockReturnValue({
      data: { user_consents_to_analytics: false },
    });

    renderHook(() => useSyncTelemetryConsent());

    expect(setTelemetryConsentMock).toHaveBeenCalledWith("denied", {
      syncToCloud: false,
    });
  });

  it("calls opt-in when user_consents_to_analytics is true", () => {
    useSettingsMock.mockReturnValue({
      data: { user_consents_to_analytics: true },
    });

    renderHook(() => useSyncTelemetryConsent());

    expect(setTelemetryConsentMock).toHaveBeenCalledWith("granted", {
      syncToCloud: false,
    });
  });

  it("does nothing while settings are still loading (data === undefined)", () => {
    useSettingsMock.mockReturnValue({ data: undefined });

    renderHook(() => useSyncTelemetryConsent());

    expect(setTelemetryConsentMock).not.toHaveBeenCalled();
  });

  it("persists a newer pre-login grant instead of applying a stale backend denial", () => {
    state.pendingConsent = "granted";
    useSettingsMock.mockReturnValue({
      data: { user_consents_to_analytics: false },
    });

    renderHook(() => useSyncTelemetryConsent());

    expect(saveSettingsMock).toHaveBeenCalledWith({
      user_consents_to_analytics: true,
    });
    expect(setTelemetryConsentMock).not.toHaveBeenCalled();
    expect(clearPendingCloudTelemetryConsentMock).not.toHaveBeenCalled();
    expect(useSaveSettingsMock).toHaveBeenCalledWith("personal", { retry: 2 });
  });

  it("clears a pending browser choice only after the backend confirms it", () => {
    state.pendingConsent = "granted";
    useSettingsMock.mockReturnValue({
      data: { user_consents_to_analytics: true },
    });

    renderHook(() => useSyncTelemetryConsent());

    expect(clearPendingCloudTelemetryConsentMock).toHaveBeenCalledWith(
      "granted",
    );
    expect(saveSettingsMock).not.toHaveBeenCalled();
    expect(setTelemetryConsentMock).not.toHaveBeenCalled();
  });

  it("keeps the choice pending when only a local backend confirms it", () => {
    state.backendKind = "local";
    state.pendingConsent = "granted";
    useSettingsMock.mockReturnValue({
      data: { user_consents_to_analytics: true },
    });

    renderHook(() => useSyncTelemetryConsent());

    expect(clearPendingCloudTelemetryConsentMock).not.toHaveBeenCalled();
    expect(saveSettingsMock).not.toHaveBeenCalled();
  });

  it("does not start another mutation while bounded retries run", () => {
    state.pendingConsent = "granted";
    useSettingsMock.mockReturnValue({
      data: { user_consents_to_analytics: false },
    });

    const { rerender } = renderHook(() => useSyncTelemetryConsent());
    state.isSavingSettings = true;
    rerender();
    state.isSavingSettings = false;
    rerender();

    expect(saveSettingsMock).toHaveBeenCalledTimes(1);
  });

  it("reacts when a first-run choice is made after the hook has mounted", () => {
    useSettingsMock.mockReturnValue({
      data: { user_consents_to_analytics: false },
    });
    renderHook(() => useSyncTelemetryConsent());
    expect(setTelemetryConsentMock).toHaveBeenCalledWith("denied", {
      syncToCloud: false,
    });

    vi.clearAllMocks();
    state.pendingConsent = "granted";
    act(() => pendingConsentListener?.());

    expect(saveSettingsMock).toHaveBeenCalledWith({
      user_consents_to_analytics: true,
    });
    expect(setTelemetryConsentMock).not.toHaveBeenCalled();
  });
});
