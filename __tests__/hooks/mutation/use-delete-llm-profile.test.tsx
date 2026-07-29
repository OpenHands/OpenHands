import { describe, expect, it, vi, afterEach, beforeEach } from "vitest";
import React from "react";
import { renderHook, waitFor, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useDeleteLlmProfile } from "#/hooks/mutation/use-delete-llm-profile";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  LLM_PROFILES_QUERY_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";

vi.mock("#/api/profiles-service/profiles-service.api");
vi.mock("#/api/settings-service/settings-service.api");

describe("useDeleteLlmProfile", () => {
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

  it("calls ProfilesService.deleteProfile with name", async () => {
    vi.mocked(ProfilesService.deleteProfile).mockResolvedValue({
      name: "old-profile",
      message: "Profile deleted",
    });

    const { result } = renderHook(() => useDeleteLlmProfile(), { wrapper });

    await act(async () => {
      await result.current.mutateAsync("old-profile");
    });

    expect(ProfilesService.deleteProfile).toHaveBeenCalledWith("old-profile");
  });

  it("invalidates LLM_PROFILES_QUERY_KEYS.all on success", async () => {
    vi.mocked(ProfilesService.deleteProfile).mockResolvedValue({
      name: "deleted-profile",
      message: "Profile deleted",
    });

    queryClient.setQueryData(LLM_PROFILES_QUERY_KEYS.all, {
      profiles: [
        {
          name: "deleted-profile",
          model: "gpt-4",
          base_url: null,
          api_key_set: true,
        },
      ],
    });
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
    const invalidateCacheSpy = vi.spyOn(SettingsService, "invalidateCache");

    const { result } = renderHook(() => useDeleteLlmProfile(), { wrapper });

    await act(async () => {
      await result.current.mutateAsync("deleted-profile");
    });

    // Verifies all three cache invalidations occur on success
    expect(invalidateCacheSpy).toHaveBeenCalled();
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: LLM_PROFILES_QUERY_KEYS.all,
    });
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: SETTINGS_QUERY_KEYS.personal(),
    });
  });

  it("clears a deleted local title profile preference", async () => {
    vi.mocked(ProfilesService.deleteProfile).mockResolvedValue({
      name: "Titles",
      message: "Profile deleted",
    });
    vi.mocked(SettingsService.getSettings).mockResolvedValue({
      title_llm_profile: "Titles",
    } as never);

    const { result } = renderHook(() => useDeleteLlmProfile(), { wrapper });

    await act(async () => {
      await result.current.mutateAsync("Titles");
    });

    expect(SettingsService.saveSettings).toHaveBeenCalledWith({
      title_llm_profile: null,
    });
  });

  it("handles delete errors", async () => {
    const error = new Error("Profile not found");
    vi.mocked(ProfilesService.deleteProfile).mockRejectedValue(error);

    const { result } = renderHook(() => useDeleteLlmProfile(), { wrapper });

    await expect(
      act(async () => {
        await result.current.mutateAsync("nonexistent-profile");
      }),
    ).rejects.toThrow("Profile not found");

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });
  });
});
