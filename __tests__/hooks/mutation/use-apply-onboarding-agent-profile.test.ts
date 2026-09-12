import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { useApplyOnboardingAgentProfile } from "#/hooks/mutation/use-apply-onboarding-agent-profile";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  default: {
    saveProfile: vi.fn(),
    getProfile: vi.fn(),
    activateProfile: vi.fn(),
  },
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME: "default",
}));

const { displayWarningToast } = vi.hoisted(() => ({
  displayWarningToast: vi.fn(),
}));
vi.mock("#/utils/custom-toast-handlers", () => ({
  displayWarningToast,
}));

function renderApplyHook() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return renderHook(() => useApplyOnboardingAgentProfile(), {
    wrapper: ({ children }) =>
      React.createElement(
        QueryClientProvider,
        { client: queryClient },
        children,
      ),
  });
}

describe("useApplyOnboardingAgentProfile", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("activates the profile by id after saving, without warning the user", async () => {
    vi.mocked(AgentProfilesService.saveProfile).mockResolvedValue({
      name: "default",
    } as never);
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      profile: { id: "profile-uuid" },
    } as never);
    vi.mocked(AgentProfilesService.activateProfile).mockResolvedValue(
      undefined as never,
    );

    const { result } = renderApplyHook();
    await result.current({
      agent_kind: "openhands",
      llm_profile_ref: "gpt-4o",
    });

    expect(AgentProfilesService.activateProfile).toHaveBeenCalledWith(
      "profile-uuid",
    );
    expect(displayWarningToast).not.toHaveBeenCalled();
  });

  it("does not block on failure, but warns the user instead of failing silently", async () => {
    // Regression: this was try/catch-swallowed to a bare console.error with
    // no user-visible signal at all — the active agent profile is left
    // stranded on its old/seeded config with zero indication anything went
    // wrong, in the one flow (first-run onboarding) where that matters most.
    vi.mocked(AgentProfilesService.saveProfile).mockRejectedValue(
      new Error("network down"),
    );

    const { result } = renderApplyHook();

    // Doesn't throw — the "never block onboarding" guarantee still holds.
    await expect(
      result.current({ agent_kind: "openhands", llm_profile_ref: "gpt-4o" }),
    ).resolves.toBeUndefined();

    await waitFor(() => {
      expect(displayWarningToast).toHaveBeenCalledWith(
        "ONBOARDING$LLM_FINALIZE_FAILED",
      );
    });
    expect(AgentProfilesService.activateProfile).not.toHaveBeenCalled();
  });
});
