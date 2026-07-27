import { describe, expect, it, vi, afterEach, beforeEach } from "vitest";
import React from "react";
import { renderHook, waitFor, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  useLlmProfiles,
  LLM_PROFILES_QUERY_KEYS,
} from "#/hooks/query/use-llm-profiles";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import type { Backend } from "#/api/backend-registry/types";

vi.mock("#/api/profiles-service/profiles-service.api");

const localBackend1: Backend = {
  id: "local-1",
  name: "Local 1",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};

const localBackend2: Backend = {
  id: "local-2",
  name: "Local 2",
  host: "http://localhost:9000",
  apiKey: "session-key-2",
  kind: "local",
};

describe("useLlmProfiles", () => {
  let queryClient: QueryClient;
  let wrapper: ({
    children,
  }: {
    children: React.ReactNode;
  }) => React.ReactElement;

  beforeEach(() => {
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend1, localBackend2]);
    setActiveSelection({ backendId: localBackend1.id, orgId: null });

    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });
    wrapper = ({ children }: { children: React.ReactNode }) =>
      React.createElement(
        QueryClientProvider,
        { client: queryClient },
        React.createElement(ActiveBackendProvider, null, children),
      );
  });

  afterEach(() => {
    queryClient.clear();
    vi.clearAllMocks();
    __resetActiveStoreForTests();
  });

  it("fetches profiles from ProfilesService.listProfiles", async () => {
    const mockProfiles = {
      profiles: [
        {
          name: "profile-1",
          model: "openai/gpt-4",
          base_url: null,
          api_key_set: true,
        },
        {
          name: "profile-2",
          model: "anthropic/claude-3",
          base_url: null,
          api_key_set: false,
        },
      ],
      active_profile: "profile-1",
    };

    vi.mocked(ProfilesService.listProfiles).mockResolvedValue(mockProfiles);

    const { result } = renderHook(() => useLlmProfiles(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(ProfilesService.listProfiles).toHaveBeenCalledTimes(1);
    expect(result.current.data).toEqual(mockProfiles);
    expect(result.current.data?.profiles).toHaveLength(2);
  });

  it("includes backend.id and orgId in query key for cache isolation", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: [],
      active_profile: null,
    });

    const { result } = renderHook(() => useLlmProfiles(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    const queries = queryClient.getQueryCache().findAll({
      queryKey: LLM_PROFILES_QUERY_KEYS.all,
    });
    expect(queries).toHaveLength(1);
    // Verify full query key includes backend.id and orgId
    expect(queries[0].queryKey).toEqual([
      ...LLM_PROFILES_QUERY_KEYS.all,
      localBackend1.id,
      null, // orgId
    ]);
  });

  it("creates separate cache entries when switching backends", async () => {
    const profilesBackend1 = {
      profiles: [
        {
          name: "profile-backend-1",
          model: "openai/gpt-4",
          base_url: null,
          api_key_set: true,
        },
      ],
      active_profile: null,
    };
    const profilesBackend2 = {
      profiles: [
        {
          name: "profile-backend-2",
          model: "anthropic/claude-3",
          base_url: null,
          api_key_set: true,
        },
      ],
      active_profile: null,
    };

    vi.mocked(ProfilesService.listProfiles)
      .mockResolvedValueOnce(profilesBackend1)
      .mockResolvedValueOnce(profilesBackend2);

    // Fetch profiles for backend 1
    const { result, rerender } = renderHook(() => useLlmProfiles(), {
      wrapper,
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });
    expect(result.current.data?.profiles[0].name).toBe("profile-backend-1");

    // Switch to backend 2
    act(() => {
      setActiveSelection({ backendId: localBackend2.id, orgId: null });
    });
    rerender();

    await waitFor(() => {
      expect(result.current.data?.profiles[0].name).toBe("profile-backend-2");
    });

    // Verify we have 2 separate cache entries
    const queries = queryClient.getQueryCache().findAll({
      queryKey: LLM_PROFILES_QUERY_KEYS.all,
    });
    expect(queries).toHaveLength(2);

    // Verify each has the correct backend.id in the query key
    const queryKeys = queries.map((q) => q.queryKey);
    expect(queryKeys).toContainEqual([
      ...LLM_PROFILES_QUERY_KEYS.all,
      localBackend1.id,
      null,
    ]);
    expect(queryKeys).toContainEqual([
      ...LLM_PROFILES_QUERY_KEYS.all,
      localBackend2.id,
      null,
    ]);
  });

  it("has staleTime of 5 minutes", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: [],
      active_profile: null,
    });

    const { result } = renderHook(() => useLlmProfiles(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    const queries = queryClient.getQueryCache().findAll({
      queryKey: LLM_PROFILES_QUERY_KEYS.all,
    });
    expect(queries).toHaveLength(1);
    expect((queries[0].options as Record<string, unknown>).staleTime).toBe(
      1000 * 60 * 5,
    );
  });

  it("has gcTime of 15 minutes", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: [],
      active_profile: null,
    });

    const { result } = renderHook(() => useLlmProfiles(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    const queries = queryClient.getQueryCache().findAll({
      queryKey: LLM_PROFILES_QUERY_KEYS.all,
    });
    expect(queries).toHaveLength(1);
    expect(queries[0].options.gcTime).toBe(1000 * 60 * 15);
  });

  it("disables toast notifications via meta", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: [],
      active_profile: null,
    });

    const { result } = renderHook(() => useLlmProfiles(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    const queries = queryClient.getQueryCache().findAll({
      queryKey: LLM_PROFILES_QUERY_KEYS.all,
    });
    expect((queries[0].options as Record<string, unknown>).meta).toEqual({
      disableToast: true,
    });
  });

  it("handles API errors gracefully", async () => {
    const error = new Error("Network error");
    vi.mocked(ProfilesService.listProfiles).mockRejectedValue(error);

    const { result } = renderHook(() => useLlmProfiles(), { wrapper });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toBe(error);
    expect(result.current.data).toBeUndefined();
  });
});
