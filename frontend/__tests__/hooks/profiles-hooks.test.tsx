import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import ProfilesService from "#/api/settings-service/profiles-service.api";
import {
  LLM_PROFILES_QUERY_KEY,
  useLlmProfiles,
} from "#/hooks/query/use-llm-profiles";
import { useSaveLlmProfile } from "#/hooks/mutation/use-save-llm-profile";
import { useDeleteLlmProfile } from "#/hooks/mutation/use-delete-llm-profile";
import { useActivateLlmProfile } from "#/hooks/mutation/use-activate-llm-profile";
import { useRenameLlmProfile } from "#/hooks/mutation/use-rename-llm-profile";

vi.mock("#/api/settings-service/profiles-service.api", () => ({
  default: {
    listProfiles: vi.fn(),
    saveProfile: vi.fn(),
    deleteProfile: vi.fn(),
    activateProfile: vi.fn(),
    renameProfile: vi.fn(),
  },
}));

// Controlled per-test so one case can flip authentication off and assert
// the query is gated.
const authState: { data: boolean } = { data: true };
vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => authState,
}));

function makeClient() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
}

function wrapperFor(client: QueryClient) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  authState.data = true;
});

describe("useLlmProfiles", () => {
  it("fetches the list via ProfilesService.listProfiles", async () => {
    const body = {
      profiles: [
        {
          name: "openai_gpt-4o",
          model: "openai/gpt-4o",
          base_url: null,
          api_key_set: true,
        },
      ],
      active_profile: "openai_gpt-4o",
    };
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue(body);

    const { result } = renderHook(() => useLlmProfiles(), {
      wrapper: wrapperFor(makeClient()),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual(body);
    expect(ProfilesService.listProfiles).toHaveBeenCalledTimes(1);
  });

  it("does not fetch while the user is unauthenticated", async () => {
    authState.data = false;

    const { result } = renderHook(() => useLlmProfiles(), {
      wrapper: wrapperFor(makeClient()),
    });
    // Give react-query a tick to settle — if it were going to fetch, it
    // would have called the service by now.
    await new Promise((r) => {
      setTimeout(r, 0);
    });
    expect(ProfilesService.listProfiles).not.toHaveBeenCalled();
    expect(result.current.fetchStatus).toBe("idle");
  });
});

describe("useSaveLlmProfile", () => {
  it("calls saveProfile and invalidates the profiles query on success", async () => {
    vi.mocked(ProfilesService.saveProfile).mockResolvedValue();
    const client = makeClient();
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");

    const { result } = renderHook(() => useSaveLlmProfile(), {
      wrapper: wrapperFor(client),
    });
    await result.current.mutateAsync({
      name: "my-profile",
      request: { include_secrets: true, llm: { model: "openai/gpt-4o" } },
    });

    expect(ProfilesService.saveProfile).toHaveBeenCalledWith("my-profile", {
      include_secrets: true,
      llm: { model: "openai/gpt-4o" },
    });
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: [LLM_PROFILES_QUERY_KEY],
    });
  });

  it("passes an empty request when none is provided", async () => {
    vi.mocked(ProfilesService.saveProfile).mockResolvedValue();

    const { result } = renderHook(() => useSaveLlmProfile(), {
      wrapper: wrapperFor(makeClient()),
    });
    await result.current.mutateAsync({ name: "snapshot" });

    expect(ProfilesService.saveProfile).toHaveBeenCalledWith("snapshot", {});
  });
});

describe("useDeleteLlmProfile", () => {
  it("invalidates both the profiles list and the settings cache", async () => {
    // Deleting the active profile clears ``llm_profiles.active`` on the
    // backend — the settings query has to refetch or the LLM page will
    // keep showing the deleted profile as in-use.
    vi.mocked(ProfilesService.deleteProfile).mockResolvedValue();
    const client = makeClient();
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");

    const { result } = renderHook(() => useDeleteLlmProfile(), {
      wrapper: wrapperFor(client),
    });
    await result.current.mutateAsync("my-profile");

    expect(ProfilesService.deleteProfile).toHaveBeenCalledWith("my-profile");
    const invalidatedKeys = invalidateSpy.mock.calls.map(
      ([arg]) => (arg as { queryKey: unknown[] }).queryKey,
    );
    expect(invalidatedKeys).toEqual(
      expect.arrayContaining([[LLM_PROFILES_QUERY_KEY], ["settings"]]),
    );
  });
});

describe("useActivateLlmProfile", () => {
  it("invalidates both the profiles list and the settings cache", async () => {
    // Activating a profile mutates agent_settings.llm on the backend — the
    // settings query has to refetch or the LLM page will render stale data.
    vi.mocked(ProfilesService.activateProfile).mockResolvedValue();
    const client = makeClient();
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");

    const { result } = renderHook(() => useActivateLlmProfile(), {
      wrapper: wrapperFor(client),
    });
    await result.current.mutateAsync("my-profile");

    expect(ProfilesService.activateProfile).toHaveBeenCalledWith("my-profile");
    const invalidatedKeys = invalidateSpy.mock.calls.map(
      ([arg]) => (arg as { queryKey: unknown[] }).queryKey,
    );
    expect(invalidatedKeys).toEqual(
      expect.arrayContaining([[LLM_PROFILES_QUERY_KEY], ["settings"]]),
    );
  });
});

describe("useRenameLlmProfile", () => {
  it("invalidates both the profiles list and the settings cache", async () => {
    // Renaming the active profile renames ``llm_profiles.active`` on the
    // backend — the settings query has to refetch or any UI surface that
    // reads the active-profile name will stay stale.
    vi.mocked(ProfilesService.renameProfile).mockResolvedValue();
    const client = makeClient();
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");

    const { result } = renderHook(() => useRenameLlmProfile(), {
      wrapper: wrapperFor(client),
    });
    await result.current.mutateAsync({ name: "old", newName: "new" });

    expect(ProfilesService.renameProfile).toHaveBeenCalledWith("old", "new");
    const invalidatedKeys = invalidateSpy.mock.calls.map(
      ([arg]) => (arg as { queryKey: unknown[] }).queryKey,
    );
    expect(invalidatedKeys).toEqual(
      expect.arrayContaining([[LLM_PROFILES_QUERY_KEY], ["settings"]]),
    );
  });
});

describe("mutation failure paths", () => {
  // onSuccess runs only on success. If a future refactor moves the
  // invalidate call into onSettled, it will fire even on failure — these
  // tests catch that regression.
  it("propagates the rejection and does not invalidate when save fails", async () => {
    vi.mocked(ProfilesService.saveProfile).mockRejectedValue(
      new Error("conflict"),
    );
    const client = makeClient();
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");

    const { result } = renderHook(() => useSaveLlmProfile(), {
      wrapper: wrapperFor(client),
    });
    await expect(result.current.mutateAsync({ name: "x" })).rejects.toThrow(
      "conflict",
    );
    expect(invalidateSpy).not.toHaveBeenCalled();
  });

  it("propagates the rejection and does not invalidate when delete fails", async () => {
    vi.mocked(ProfilesService.deleteProfile).mockRejectedValue(
      new Error("boom"),
    );
    const client = makeClient();
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");

    const { result } = renderHook(() => useDeleteLlmProfile(), {
      wrapper: wrapperFor(client),
    });
    await expect(result.current.mutateAsync("x")).rejects.toThrow("boom");
    expect(invalidateSpy).not.toHaveBeenCalled();
  });

  it("propagates the rejection and does not invalidate when activate fails", async () => {
    vi.mocked(ProfilesService.activateProfile).mockRejectedValue(
      new Error("missing"),
    );
    const client = makeClient();
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");

    const { result } = renderHook(() => useActivateLlmProfile(), {
      wrapper: wrapperFor(client),
    });
    await expect(result.current.mutateAsync("x")).rejects.toThrow("missing");
    expect(invalidateSpy).not.toHaveBeenCalled();
  });

  it("propagates the rejection and does not invalidate when rename fails", async () => {
    vi.mocked(ProfilesService.renameProfile).mockRejectedValue(
      new Error("exists"),
    );
    const client = makeClient();
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");

    const { result } = renderHook(() => useRenameLlmProfile(), {
      wrapper: wrapperFor(client),
    });
    await expect(
      result.current.mutateAsync({ name: "a", newName: "b" }),
    ).rejects.toThrow("exists");
    expect(invalidateSpy).not.toHaveBeenCalled();
  });
});
