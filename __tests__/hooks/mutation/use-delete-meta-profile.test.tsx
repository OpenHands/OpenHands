import { describe, expect, it, vi, afterEach, beforeEach } from "vitest";
import React from "react";
import { renderHook, waitFor, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useDeleteMetaProfile } from "#/hooks/mutation/use-delete-meta-profile";
import MetaProfilesService from "#/api/meta-profiles-service/meta-profiles-service.api";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  META_PROFILES_QUERY_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";

vi.mock("#/api/meta-profiles-service/meta-profiles-service.api");
vi.mock("#/api/settings-service/settings-service.api");
vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  __esModule: true,
  default: { listProfiles: vi.fn(), getProfile: vi.fn(), saveProfile: vi.fn() },
}));

const openHands = (name: string, router: string | null) => ({
  id: `id-${name}`,
  name,
  revision: 1,
  agent_kind: "openhands",
  llm_profile_ref: "llm",
  mcp_server_refs: null,
  meta_profile_ref: router,
  enable_classify_and_switch_llm_tool: router !== null,
});

describe("useDeleteMetaProfile", () => {
  let queryClient: QueryClient;
  let wrapper: ({
    children,
  }: {
    children: React.ReactNode;
  }) => React.ReactElement;

  beforeEach(() => {
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: [],
      active_agent_profile_id: null,
    } as never);
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

  it("detaches the router from the local agent profiles that use it, first", async () => {
    const profiles = [
      openHands("uses-it", "balanced"),
      openHands("other-router", "cheap"),
    ];
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: [
        ...profiles,
        { id: "id-acp", name: "acp", agent_kind: "acp", revision: 1 },
      ],
      active_agent_profile_id: "id-uses-it",
    } as never);
    vi.mocked(AgentProfilesService.getProfile).mockImplementation(
      async (name) =>
        ({
          name,
          profile: profiles.find((profile) => profile.name === name),
        }) as never,
    );
    vi.mocked(MetaProfilesService.deleteMetaProfile).mockImplementation(
      async () => {
        expect(AgentProfilesService.saveProfile).toHaveBeenCalledTimes(1);
        return { name: "balanced", message: "deleted" };
      },
    );

    const { result } = renderHook(() => useDeleteMetaProfile(), { wrapper });
    await act(async () => {
      await result.current.mutateAsync("balanced");
    });

    expect(AgentProfilesService.saveProfile).toHaveBeenCalledWith(
      "uses-it",
      expect.objectContaining({
        meta_profile_ref: null,
        enable_classify_and_switch_llm_tool: false,
        llm_profile_ref: "llm",
      }),
    );
    expect(MetaProfilesService.deleteMetaProfile).toHaveBeenCalledWith(
      "balanced",
    );
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
