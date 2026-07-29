import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const useSettingsMock = vi.fn();
vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => useSettingsMock(),
}));

const useActiveBackendMock = vi.fn();
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => useActiveBackendMock(),
}));

const useCloudCurrentUserIdMock = vi.fn();
vi.mock("#/hooks/query/use-cloud-current-user-id", () => ({
  useCloudCurrentUserId: () => useCloudCurrentUserIdMock(),
}));

const setTelemetryCloudContextMock = vi.fn();
const setTelemetryIdentityMock = vi.fn();
vi.mock("#/services/telemetry", () => ({
  setTelemetryCloudContext: (...args: unknown[]) =>
    setTelemetryCloudContextMock(...args),
  setTelemetryIdentity: (...args: unknown[]) =>
    setTelemetryIdentityMock(...args),
}));

const trackBackendAddedMock = vi.fn();
vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({ trackBackendAdded: trackBackendAddedMock }),
}));

import { useTelemetryIdentity } from "#/hooks/use-telemetry-identity";

const BACKEND_ID = "cloud-1";
const cloudBackend = { kind: "cloud" as const, id: BACKEND_ID };
const cookieCloudBackend = {
  kind: "cloud" as const,
  id: BACKEND_ID,
  authMode: "cookie" as const,
};
const localBackend = { kind: "local" as const, id: "local-1" };

describe("useTelemetryIdentity", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.sessionStorage.clear();
    useActiveBackendMock.mockReturnValue({
      backend: cloudBackend,
      orgId: "org-123",
    });
    useSettingsMock.mockReturnValue({
      data: { email: "user@example.com", git_user_email: "git@example.com" },
    });
    useCloudCurrentUserIdMock.mockReturnValue({
      [BACKEND_ID]: { userId: "user-123", isLoading: false },
    });
  });

  it("declares the current Cloud user context", () => {
    renderHook(() => useTelemetryIdentity());

    expect(setTelemetryCloudContextMock).toHaveBeenCalledWith({
      userId: "user-123",
      email: "user@example.com",
      orgId: "org-123",
    });
    expect(setTelemetryIdentityMock).toHaveBeenCalledWith("user-123", {
      email: "user@example.com",
    });
  });

  it("falls back to the git email and omits an absent email", () => {
    useSettingsMock.mockReturnValue({
      data: { email: "", git_user_email: "git@example.com" },
    });
    const { rerender } = renderHook(() => useTelemetryIdentity());
    expect(setTelemetryCloudContextMock).toHaveBeenLastCalledWith({
      userId: "user-123",
      email: "git@example.com",
      orgId: "org-123",
    });

    useSettingsMock.mockReturnValue({ data: {} });
    rerender();
    expect(setTelemetryCloudContextMock).toHaveBeenLastCalledWith({
      userId: "user-123",
      email: undefined,
      orgId: "org-123",
    });
  });

  it("preserves Cloud user context while the identity query loads", () => {
    useCloudCurrentUserIdMock.mockReturnValue({
      [BACKEND_ID]: { userId: null, isLoading: true },
    });

    renderHook(() => useTelemetryIdentity());

    expect(setTelemetryCloudContextMock).not.toHaveBeenCalled();
  });

  it("declares logout only after the Cloud identity query settles", () => {
    useCloudCurrentUserIdMock.mockReturnValue({
      [BACKEND_ID]: { userId: null, isLoading: false },
    });

    renderHook(() => useTelemetryIdentity());

    expect(setTelemetryCloudContextMock).toHaveBeenCalledWith(null);
    expect(setTelemetryIdentityMock).toHaveBeenCalledWith(null);
  });

  it("clears Cloud user context while a local backend is active", () => {
    useActiveBackendMock.mockReturnValue({
      backend: localBackend,
      orgId: null,
    });

    renderHook(() => useTelemetryIdentity());

    expect(setTelemetryCloudContextMock).toHaveBeenCalledWith(null);
    expect(setTelemetryIdentityMock).not.toHaveBeenCalled();
  });

  it("declares a changed Cloud account", () => {
    const { rerender } = renderHook(() => useTelemetryIdentity());
    useCloudCurrentUserIdMock.mockReturnValue({
      [BACKEND_ID]: { userId: "user-456", isLoading: false },
    });

    rerender();

    expect(setTelemetryCloudContextMock).toHaveBeenLastCalledWith({
      userId: "user-456",
      email: "user@example.com",
      orgId: "org-123",
    });
    expect(setTelemetryIdentityMock).toHaveBeenLastCalledWith("user-456", {
      email: "user@example.com",
    });
  });

  it("redeclares the user after Cloud connection credentials change", () => {
    const { rerender } = renderHook(() => useTelemetryIdentity());
    vi.clearAllMocks();
    useActiveBackendMock.mockReturnValue({
      backend: { ...cloudBackend, connectionRevision: 1 },
      orgId: "org-123",
    });

    rerender();

    expect(setTelemetryIdentityMock).toHaveBeenCalledWith("user-123", {
      email: "user@example.com",
    });
  });

  it("tracks cookie-auth Cloud backend addition after identity and email resolve", () => {
    useActiveBackendMock.mockReturnValue({
      backend: cookieCloudBackend,
      orgId: "org-123",
    });

    renderHook(() => useTelemetryIdentity());

    expect(trackBackendAddedMock).toHaveBeenCalledWith({
      backendKind: "cloud",
      connectionMethod: "cloud_cookie",
      hasApiKey: false,
      source: "cloud_auto_connect",
    });
  });

  it("dedupes cookie-auth Cloud backend additions within a browser session", () => {
    useActiveBackendMock.mockReturnValue({
      backend: cookieCloudBackend,
      orgId: "org-123",
    });

    const { rerender } = renderHook(() => useTelemetryIdentity());
    rerender();

    expect(trackBackendAddedMock).toHaveBeenCalledTimes(1);
  });

  it("waits for an email before tracking cookie-auth Cloud backend addition", () => {
    useActiveBackendMock.mockReturnValue({
      backend: cookieCloudBackend,
      orgId: "org-123",
    });
    useSettingsMock.mockReturnValue({ data: {} });
    const { rerender } = renderHook(() => useTelemetryIdentity());

    expect(trackBackendAddedMock).not.toHaveBeenCalled();

    useSettingsMock.mockReturnValue({
      data: { email: "", git_user_email: "git@example.com" },
    });
    rerender();

    expect(trackBackendAddedMock).toHaveBeenCalledWith(
      expect.objectContaining({ connectionMethod: "cloud_cookie" }),
    );
  });

  it("dedupes cookie-auth backend additions when sessionStorage writes fail", () => {
    const storageFailureCloudBackend = {
      kind: "cloud" as const,
      id: "cloud-storage-failure",
      authMode: "cookie" as const,
    };
    useActiveBackendMock.mockReturnValue({
      backend: storageFailureCloudBackend,
      orgId: "org-123",
    });
    useCloudCurrentUserIdMock.mockReturnValue({
      [storageFailureCloudBackend.id]: {
        userId: "storage-failure-user",
        isLoading: false,
      },
    });
    const setItemSpy = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(() => {
        throw new Error("sessionStorage writes blocked");
      });

    try {
      const { rerender } = renderHook(() => useTelemetryIdentity());
      rerender();

      expect(trackBackendAddedMock).toHaveBeenCalledTimes(1);
    } finally {
      setItemSpy.mockRestore();
    }
  });

  it("does not track backend_added for non-cookie Cloud identity resolution", () => {
    renderHook(() => useTelemetryIdentity());

    expect(trackBackendAddedMock).not.toHaveBeenCalled();
  });
});
