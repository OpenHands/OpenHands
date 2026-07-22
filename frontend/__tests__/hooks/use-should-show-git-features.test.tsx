import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { useShouldShowGitFeatures } from "#/hooks/use-should-show-git-features";
import * as useConfigModule from "#/hooks/query/use-config";
import * as useIsAuthedModule from "#/hooks/query/use-is-authed";
import * as useUserProvidersModule from "#/hooks/use-user-providers";

vi.mock("#/hooks/query/use-config");
vi.mock("#/hooks/query/use-is-authed");
vi.mock("#/hooks/use-user-providers");

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe("useShouldShowGitFeatures", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should return false when config is not loaded", () => {
    vi.mocked(useConfigModule.useConfig).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    } as any);
    vi.mocked(useIsAuthedModule.useIsAuthed).mockReturnValue({
      data: true,
    } as any);
    vi.mocked(useUserProvidersModule.useUserProviders).mockReturnValue({
      providers: ["github"],
      isLoadingSettings: false,
    });

    const { result } = renderHook(() => useShouldShowGitFeatures(), {
      wrapper: createWrapper(),
    });

    expect(result.current).toBe(false);
  });

  it("should return false when not authenticated", () => {
    vi.mocked(useConfigModule.useConfig).mockReturnValue({
      data: { app_mode: "saas" },
      isLoading: false,
      error: null,
    } as any);
    vi.mocked(useIsAuthedModule.useIsAuthed).mockReturnValue({
      data: false,
    } as any);
    vi.mocked(useUserProvidersModule.useUserProviders).mockReturnValue({
      providers: ["github"],
      isLoadingSettings: false,
    });

    const { result } = renderHook(() => useShouldShowGitFeatures(), {
      wrapper: createWrapper(),
    });

    expect(result.current).toBe(false);
  });

  it("should return false when no providers are configured in SaaS mode", () => {
    vi.mocked(useConfigModule.useConfig).mockReturnValue({
      data: { app_mode: "saas" },
      isLoading: false,
      error: null,
    } as any);
    vi.mocked(useIsAuthedModule.useIsAuthed).mockReturnValue({
      data: true,
    } as any);
    vi.mocked(useUserProvidersModule.useUserProviders).mockReturnValue({
      providers: [],
      isLoadingSettings: false,
    });

    const { result } = renderHook(() => useShouldShowGitFeatures(), {
      wrapper: createWrapper(),
    });

    expect(result.current).toBe(false);
  });

  it("should return false when no providers are configured in OSS mode", () => {
    vi.mocked(useConfigModule.useConfig).mockReturnValue({
      data: { app_mode: "oss" },
      isLoading: false,
      error: null,
    } as any);
    vi.mocked(useIsAuthedModule.useIsAuthed).mockReturnValue({
      data: true,
    } as any);
    vi.mocked(useUserProvidersModule.useUserProviders).mockReturnValue({
      providers: [],
      isLoadingSettings: false,
    });

    const { result } = renderHook(() => useShouldShowGitFeatures(), {
      wrapper: createWrapper(),
    });

    expect(result.current).toBe(false);
  });

  it("should return true when authenticated with a provider in SaaS mode", () => {
    vi.mocked(useConfigModule.useConfig).mockReturnValue({
      data: { app_mode: "saas" },
      isLoading: false,
      error: null,
    } as any);
    vi.mocked(useIsAuthedModule.useIsAuthed).mockReturnValue({
      data: true,
    } as any);
    vi.mocked(useUserProvidersModule.useUserProviders).mockReturnValue({
      providers: ["github"],
      isLoadingSettings: false,
    });

    const { result } = renderHook(() => useShouldShowGitFeatures(), {
      wrapper: createWrapper(),
    });

    expect(result.current).toBe(true);
  });

  it("should return true when authenticated with a provider in OSS mode", () => {
    vi.mocked(useConfigModule.useConfig).mockReturnValue({
      data: { app_mode: "oss" },
      isLoading: false,
      error: null,
    } as any);
    vi.mocked(useIsAuthedModule.useIsAuthed).mockReturnValue({
      data: true,
    } as any);
    vi.mocked(useUserProvidersModule.useUserProviders).mockReturnValue({
      providers: ["gitlab"],
      isLoadingSettings: false,
    });

    const { result } = renderHook(() => useShouldShowGitFeatures(), {
      wrapper: createWrapper(),
    });

    expect(result.current).toBe(true);
  });

  it("should return true when multiple providers are configured", () => {
    vi.mocked(useConfigModule.useConfig).mockReturnValue({
      data: { app_mode: "saas" },
      isLoading: false,
      error: null,
    } as any);
    vi.mocked(useIsAuthedModule.useIsAuthed).mockReturnValue({
      data: true,
    } as any);
    vi.mocked(useUserProvidersModule.useUserProviders).mockReturnValue({
      providers: ["github", "gitlab"],
      isLoadingSettings: false,
    });

    const { result } = renderHook(() => useShouldShowGitFeatures(), {
      wrapper: createWrapper(),
    });

    expect(result.current).toBe(true);
  });
});
