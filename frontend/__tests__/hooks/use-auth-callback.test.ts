import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mockUseConfig = vi.fn();
const mockUseIsAuthed = vi.fn();
const mockUseSettings = vi.fn();
const mockNavigate = vi.fn();
const mockSetLoginMethod = vi.fn();
const mockUseLocation = vi.fn();

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => mockUseConfig(),
}));

vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => mockUseIsAuthed(),
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => mockUseSettings(),
}));

vi.mock("react-router", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-router")>()),
  useNavigate: () => mockNavigate,
  useLocation: () => mockUseLocation(),
}));

vi.mock("#/utils/local-storage", () => ({
  LoginMethod: {
    GITHUB: "github",
    GITLAB: "gitlab",
    BITBUCKET: "bitbucket",
    BITBUCKET_DATA_CENTER: "bitbucket_data_center",
    AZURE_DEVOPS: "azure_devops",
    ENTERPRISE_SSO: "enterprise_sso",
  },
  setLoginMethod: (method: string) => mockSetLoginMethod(method),
}));

import { useAuthCallback } from "#/hooks/use-auth-callback";

const SAAS_CONFIG = { app_mode: "saas" };
const OSS_CONFIG = { app_mode: "oss" };

const makeLocation = (search: string, pathname = "/") => ({
  search,
  pathname,
});

describe("useAuthCallback", () => {
  beforeEach(() => {
    mockUseConfig.mockReturnValue({ data: SAAS_CONFIG });
    mockUseIsAuthed.mockReturnValue({ data: true, isLoading: false });
    mockUseSettings.mockReturnValue({ data: { stay_logged_in: true } });
    mockUseLocation.mockReturnValue(
      makeLocation("?login_method=github&returnTo=%2Fdashboard"),
    );
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should not set login method when app_mode is not saas", () => {
    mockUseConfig.mockReturnValue({ data: OSS_CONFIG });

    renderHook(() => useAuthCallback());

    expect(mockSetLoginMethod).not.toHaveBeenCalled();
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it("should not set login method while auth is loading", () => {
    mockUseIsAuthed.mockReturnValue({ data: undefined, isLoading: true });

    renderHook(() => useAuthCallback());

    expect(mockSetLoginMethod).not.toHaveBeenCalled();
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it("should not set login method when user is not authenticated", () => {
    mockUseIsAuthed.mockReturnValue({ data: false, isLoading: false });

    renderHook(() => useAuthCallback());

    expect(mockSetLoginMethod).not.toHaveBeenCalled();
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it("should set login method when authenticated and stay_logged_in is true", () => {
    mockUseSettings.mockReturnValue({ data: { stay_logged_in: true } });

    renderHook(() => useAuthCallback());

    expect(mockSetLoginMethod).toHaveBeenCalledWith("github");
  });

  it("should NOT set login method when stay_logged_in is false", () => {
    mockUseSettings.mockReturnValue({ data: { stay_logged_in: false } });

    renderHook(() => useAuthCallback());

    expect(mockSetLoginMethod).not.toHaveBeenCalled();
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it("should set login method when settings data is undefined (stay_logged_in not set)", () => {
    // settings?.stay_logged_in is undefined when data is undefined
    // undefined !== false is true, so login method should be stored
    mockUseSettings.mockReturnValue({ data: undefined });

    renderHook(() => useAuthCallback());

    expect(mockSetLoginMethod).toHaveBeenCalledWith("github");
  });

  it("should set login method when settings is null (stay_logged_in not set)", () => {
    // null?.stay_logged_in is undefined, and undefined !== false, so should proceed
    mockUseSettings.mockReturnValue({ data: null });

    renderHook(() => useAuthCallback());

    expect(mockSetLoginMethod).toHaveBeenCalledWith("github");
  });

  it("should navigate to returnTo path after setting login method", () => {
    mockUseLocation.mockReturnValue(
      makeLocation("?login_method=github&returnTo=%2Fdashboard"),
    );

    renderHook(() => useAuthCallback());

    expect(mockNavigate).toHaveBeenCalledWith(
      "/dashboard",
      expect.objectContaining({ replace: true }),
    );
  });

  it("should navigate to '/' when returnTo is '/login'", () => {
    mockUseLocation.mockReturnValue(
      makeLocation("?login_method=github&returnTo=%2Flogin"),
    );

    renderHook(() => useAuthCallback());

    expect(mockSetLoginMethod).toHaveBeenCalledWith("github");
    expect(mockNavigate).toHaveBeenCalledWith(
      "/",
      expect.objectContaining({ replace: true }),
    );
  });

  it("should not set login method when login_method param is missing", () => {
    mockUseLocation.mockReturnValue(makeLocation(""));

    renderHook(() => useAuthCallback());

    expect(mockSetLoginMethod).not.toHaveBeenCalled();
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it("should not set login method when login_method param is invalid", () => {
    mockUseLocation.mockReturnValue(
      makeLocation("?login_method=invalid_method"),
    );

    renderHook(() => useAuthCallback());

    expect(mockSetLoginMethod).not.toHaveBeenCalled();
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it("should navigate to '/' when no returnTo is provided", () => {
    mockUseLocation.mockReturnValue(makeLocation("?login_method=github"));

    renderHook(() => useAuthCallback());

    expect(mockSetLoginMethod).toHaveBeenCalledWith("github");
    expect(mockNavigate).toHaveBeenCalledWith(
      "/",
      expect.objectContaining({ replace: true }),
    );
  });

  it("should not navigate when stay_logged_in is false even if valid login_method is provided", () => {
    mockUseSettings.mockReturnValue({ data: { stay_logged_in: false } });

    renderHook(() => useAuthCallback());

    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it("should handle all valid LoginMethod values when stay_logged_in is true", () => {
    const validMethods = [
      "github",
      "gitlab",
      "bitbucket",
      "bitbucket_data_center",
      "azure_devops",
      "enterprise_sso",
    ];

    for (const method of validMethods) {
      vi.clearAllMocks();
      mockUseLocation.mockReturnValue(
        makeLocation(`?login_method=${method}&returnTo=%2Fdashboard`),
      );

      renderHook(() => useAuthCallback());

      expect(mockSetLoginMethod).toHaveBeenCalledWith(method);
    }
  });

  it("should use current pathname as destination when no returnTo and pathname is not root or login", () => {
    mockUseLocation.mockReturnValue(
      makeLocation("?login_method=github", "/some/page"),
    );

    renderHook(() => useAuthCallback());

    expect(mockNavigate).toHaveBeenCalledWith(
      "/some/page",
      expect.objectContaining({ replace: true }),
    );
  });
});