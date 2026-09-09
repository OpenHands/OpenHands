import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useEnsureActiveProfile } from "#/hooks/use-ensure-active-profile";

const mockActivate = vi.fn();
const mockUseActiveBackend = vi.fn();
const mockUseLlmProfiles = vi.fn();

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => mockUseActiveBackend(),
}));
vi.mock("#/hooks/query/use-llm-profiles", () => ({
  useLlmProfiles: () => mockUseLlmProfiles(),
}));
vi.mock("#/hooks/mutation/use-activate-llm-profile", () => ({
  useActivateLlmProfile: () => ({ mutate: mockActivate, isPending: false }),
}));

const local = { backend: { id: "b1", kind: "local" }, orgId: null };
const cloud = { backend: { id: "c1", kind: "cloud" }, orgId: null };
const profile = (name: string, api_key_set = true) => ({
  name,
  model: "m",
  base_url: null,
  api_key_set,
});

describe("useEnsureActiveProfile", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseActiveBackend.mockReturnValue(local);
  });

  it("activates the first profile when none is active", () => {
    mockUseLlmProfiles.mockReturnValue({
      data: { profiles: [profile("a"), profile("b")], active_profile: null },
    });
    renderHook(() => useEnsureActiveProfile());
    expect(mockActivate).toHaveBeenCalledWith("a");
  });

  it("activates a remaining profile when the active one is dangling", () => {
    mockUseLlmProfiles.mockReturnValue({
      data: { profiles: [profile("b")], active_profile: "deleted" },
    });
    renderHook(() => useEnsureActiveProfile());
    expect(mockActivate).toHaveBeenCalledWith("b");
  });

  it("prefers a profile that has a key over an earlier keyless one", () => {
    mockUseLlmProfiles.mockReturnValue({
      data: {
        profiles: [profile("nokey", false), profile("withkey")],
        active_profile: null,
      },
    });
    renderHook(() => useEnsureActiveProfile());
    expect(mockActivate).toHaveBeenCalledWith("withkey");
  });

  it("promotes a connection-linked profile over an unusable active default", () => {
    mockUseLlmProfiles.mockReturnValue({
      data: {
        profiles: [
          profile("default", false),
          {
            name: "local-ollama",
            model: "ollama/llama3.2",
            base_url: null,
            api_key_set: false,
            provider_connection_id: "conn-ollama",
          },
        ],
        active_profile: "default",
      },
    });
    renderHook(() => useEnsureActiveProfile());
    expect(mockActivate).toHaveBeenCalledWith("local-ollama");
  });

  it("does nothing when the active profile is valid", () => {
    mockUseLlmProfiles.mockReturnValue({
      data: { profiles: [profile("a")], active_profile: "a" },
    });
    renderHook(() => useEnsureActiveProfile());
    expect(mockActivate).not.toHaveBeenCalled();
  });

  it("does nothing when there are no profiles", () => {
    mockUseLlmProfiles.mockReturnValue({
      data: { profiles: [], active_profile: "dangling" },
    });
    renderHook(() => useEnsureActiveProfile());
    expect(mockActivate).not.toHaveBeenCalled();
  });

  it("does nothing on a cloud backend", () => {
    mockUseActiveBackend.mockReturnValue(cloud);
    mockUseLlmProfiles.mockReturnValue({
      data: { profiles: [profile("a")], active_profile: null },
    });
    renderHook(() => useEnsureActiveProfile());
    expect(mockActivate).not.toHaveBeenCalled();
  });
});
