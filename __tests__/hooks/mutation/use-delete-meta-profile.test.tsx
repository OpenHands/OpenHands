import { describe, expect, it, vi, afterEach, beforeEach } from "vitest";
import React from "react";
import { renderHook, waitFor, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useDeleteMetaProfile } from "#/hooks/mutation/use-delete-meta-profile";
import MetaProfilesService from "#/api/meta-profiles-service/meta-profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  META_PROFILES_QUERY_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";

vi.mock("#/api/meta-profiles-service/meta-profiles-service.api");
vi.mock("#/api/settings-service/settings-service.api");

describe("useDeleteMetaProfile", () => {
  let queryClient: QueryClient;
  let wrapper: ({
    children,
  }: {
    children: React.ReactNode;
  }) => React.ReactElement;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    wrapper = ({ children }: { children: React.ReactNode }) =>
      React.createElement(
        QueryClientProvider,
        { client: queryClient },
        children,
      );
  });

  afterEach(() => {
    queryClient.clear();
    vi.clearAllMocks();
  });

  it("calls MetaProfilesService.deleteMetaProfile with name", async () => {
    vi.mocked(MetaProfilesService.deleteMetaProfile).mockResolvedValue({
      name: "balanced",
      message: "deleted",
    });

    const { result } = renderHook(() => useDeleteMetaProfile(), { wrapper });

    await act(async () => {
      await result.current.mutateAsync("balanced");
    });

    expect(MetaProfilesService.deleteMetaProfile).toHaveBeenCalledWith(
      "balanced",
    );
  });

  it("invalidates the meta-profile AND settings caches on success", async () => {
    // Deleting the active meta-profile clears active_meta_profile in settings,
    // so the settings caches must be refreshed too (mirrors activation).
    vi.mocked(MetaProfilesService.deleteMetaProfile).mockResolvedValue({
      name: "balanced",
      message: "deleted",
    });
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
    const invalidateCacheSpy = vi.spyOn(SettingsService, "invalidateCache");

    const { result } = renderHook(() => useDeleteMetaProfile(), { wrapper });

    await act(async () => {
      await result.current.mutateAsync("balanced");
    });

    expect(invalidateCacheSpy).toHaveBeenCalled();
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: META_PROFILES_QUERY_KEYS.all,
    });
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: SETTINGS_QUERY_KEYS.personal(),
    });
  });

  it("propagates delete errors", async () => {
    vi.mocked(MetaProfilesService.deleteMetaProfile).mockRejectedValue(
      new Error("nope"),
    );

    const { result } = renderHook(() => useDeleteMetaProfile(), { wrapper });

    await expect(
      act(async () => {
        await result.current.mutateAsync("balanced");
      }),
    ).rejects.toThrow("nope");

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });
  });
});
