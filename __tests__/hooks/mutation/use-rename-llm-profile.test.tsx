import { describe, expect, it, vi, afterEach, beforeEach } from "vitest";
import React from "react";
import { renderHook, waitFor, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useRenameLlmProfile } from "#/hooks/mutation/use-rename-llm-profile";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  LLM_PROFILES_QUERY_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";

vi.mock("#/api/profiles-service/profiles-service.api");
vi.mock("#/api/settings-service/settings-service.api");

describe("useRenameLlmProfile", () => {
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

  it("calls ProfilesService.renameProfile with name and newName", async () => {
    vi.mocked(ProfilesService.renameProfile).mockResolvedValue({
      name: "new-name",
      message: "Profile renamed",
    });

    const { result } = renderHook(() => useRenameLlmProfile(), { wrapper });

    await act(async () => {
      await result.current.mutateAsync({
        name: "old-name",
        newName: "new-name",
      });
    });

    expect(ProfilesService.renameProfile).toHaveBeenCalledWith(
      "old-name",
      "new-name",
    );
  });

  it("invalidates LLM_PROFILES_QUERY_KEYS.all on success", async () => {
    vi.mocked(ProfilesService.renameProfile).mockResolvedValue({
      name: "renamed-profile",
      message: "Profile renamed",
    });

    queryClient.setQueryData(LLM_PROFILES_QUERY_KEYS.all, {
      profiles: [
        { name: "old-name", model: "gpt-4", base_url: null, api_key_set: true },
      ],
    });
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");

    const { result } = renderHook(() => useRenameLlmProfile(), { wrapper });

    await act(async () => {
      await result.current.mutateAsync({
        name: "old-name",
        newName: "renamed-profile",
      });
    });

    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: LLM_PROFILES_QUERY_KEYS.all,
    });
  });

  it("invalidates settings queries on success", async () => {
    vi.mocked(ProfilesService.renameProfile).mockResolvedValue({
      name: "new-name",
      message: "Profile renamed",
    });

    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");

    const { result } = renderHook(() => useRenameLlmProfile(), { wrapper });

    await act(async () => {
      await result.current.mutateAsync({
        name: "old-name",
        newName: "new-name",
      });
    });

    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: SETTINGS_QUERY_KEYS.personal(),
    });
  });

  it("renames a selected local title profile preference", async () => {
    vi.mocked(ProfilesService.renameProfile).mockResolvedValue({
      name: "Titles-v2",
      message: "Profile renamed",
    });
    vi.mocked(SettingsService.getSettings).mockResolvedValue({
      title_llm_profile: "Titles",
    } as never);

    const { result } = renderHook(() => useRenameLlmProfile(), { wrapper });

    await act(async () => {
      await result.current.mutateAsync({
        name: "Titles",
        newName: "Titles-v2",
      });
    });

    expect(SettingsService.saveSettings).toHaveBeenCalledWith({
      title_llm_profile: "Titles-v2",
    });
  });

  it("handles rename errors (e.g., name conflict)", async () => {
    const error = new Error("Profile name already exists");
    vi.mocked(ProfilesService.renameProfile).mockRejectedValue(error);

    const { result } = renderHook(() => useRenameLlmProfile(), { wrapper });

    await expect(
      act(async () => {
        await result.current.mutateAsync({
          name: "old-name",
          newName: "existing-name",
        });
      }),
    ).rejects.toThrow("Profile name already exists");

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });
  });

  it("handles rename errors (e.g., profile not found)", async () => {
    const error = new Error("Profile not found");
    vi.mocked(ProfilesService.renameProfile).mockRejectedValue(error);

    const { result } = renderHook(() => useRenameLlmProfile(), { wrapper });

    await expect(
      act(async () => {
        await result.current.mutateAsync({
          name: "nonexistent",
          newName: "new-name",
        });
      }),
    ).rejects.toThrow("Profile not found");

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });
  });

  it("invalidates SettingsService cache on success", async () => {
    vi.mocked(ProfilesService.renameProfile).mockResolvedValue({
      name: "new-name",
      message: "Profile renamed",
    });

    const invalidateCacheSpy = vi.spyOn(SettingsService, "invalidateCache");

    const { result } = renderHook(() => useRenameLlmProfile(), { wrapper });

    await act(async () => {
      await result.current.mutateAsync({
        name: "old-name",
        newName: "new-name",
      });
    });

    expect(invalidateCacheSpy).toHaveBeenCalled();
  });
});
