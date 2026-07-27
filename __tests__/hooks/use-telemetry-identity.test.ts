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

import { useTelemetryIdentity } from "#/hooks/use-telemetry-identity";

const BACKEND_ID = "cloud-1";
const cloudBackend = { kind: "cloud" as const, id: BACKEND_ID };
const localBackend = { kind: "local" as const, id: "local-1" };

describe("useTelemetryIdentity", () => {
  beforeEach(() => {
    vi.clearAllMocks();
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
});
