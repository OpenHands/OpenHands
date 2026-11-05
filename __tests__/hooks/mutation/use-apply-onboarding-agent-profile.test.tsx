import { renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { HttpError } from "@openhands/typescript-client";
import { DEFAULT_ENABLED_SKILL_NAMES } from "@openhands/extensions/skills";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import * as activeBackendContext from "#/contexts/active-backend-context";
import { useApplyOnboardingAgentProfile } from "#/hooks/mutation/use-apply-onboarding-agent-profile";
import { ensureDefaultAgentProfileMigrated } from "#/hooks/use-migrate-default-agent-profile";

vi.mock("#/hooks/use-migrate-default-agent-profile", () => ({
  ensureDefaultAgentProfileMigrated: vi.fn(),
}));

vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  __esModule: true,
  default: {
    getProfile: vi.fn(),
    saveProfile: vi.fn(),
    activateProfile: vi.fn(),
  },
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME: "default",
}));

const STORED_DEFAULT = {
  id: "profile-default",
  name: "default",
  revision: 2,
  agent_kind: "openhands",
  llm_profile_ref: "old-llm",
  mcp_server_refs: ["linear"],
  public_skills: ["github"],
};

function renderApply() {
  const queryClient = new QueryClient();
  return renderHook(() => useApplyOnboardingAgentProfile(), {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  }).result.current;
}

describe("useApplyOnboardingAgentProfile", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    vi.spyOn(activeBackendContext, "useActiveBackend").mockReturnValue({
      backend: {
        id: "local-1",
        name: "Local",
        host: "http://127.0.0.1:18000",
        apiKey: "",
        kind: "local",
      },
      orgId: null,
    });
    vi.mocked(ensureDefaultAgentProfileMigrated).mockResolvedValue(true);
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      name: "default",
      profile: STORED_DEFAULT,
    } as never);
    vi.mocked(AgentProfilesService.saveProfile).mockResolvedValue({} as never);
    vi.mocked(AgentProfilesService.activateProfile).mockResolvedValue(
      {} as never,
    );
  });

  it("keeps the stored `default` settings when onboarding picks an LLM", async () => {
    await renderApply()({
      agent_kind: "openhands",
      llm_profile_ref: "new-llm",
    });

    expect(AgentProfilesService.saveProfile).toHaveBeenCalledWith("default", {
      agent_kind: "openhands",
      llm_profile_ref: "new-llm",
      mcp_server_refs: ["linear"],
      public_skills: ["github"],
    });
    expect(AgentProfilesService.activateProfile).toHaveBeenCalledWith(
      "profile-default",
    );
  });

  it("replaces `default` when onboarding picks an ACP agent", async () => {
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      name: "default",
      profile: {
        id: "profile-default",
        name: "default",
        revision: 2,
        agent_kind: "acp",
        acp_server: "custom",
        acp_command: "my-acp --flag",
      },
    } as never);

    await renderApply()({
      agent_kind: "acp",
      acp_server: "claude-code",
      acp_model: null,
    });

    expect(AgentProfilesService.saveProfile).toHaveBeenCalledWith("default", {
      agent_kind: "acp",
      acp_server: "claude-code",
      acp_model: null,
    });
  });

  it("configures `default` only after the default migration", async () => {
    let finishMigration: (value: true) => void = () => {};
    vi.mocked(ensureDefaultAgentProfileMigrated).mockReturnValue(
      new Promise((resolve) => {
        finishMigration = resolve;
      }),
    );

    const applied = renderApply()({
      agent_kind: "openhands",
      llm_profile_ref: "new-llm",
    });
    await Promise.resolve();
    expect(AgentProfilesService.getProfile).not.toHaveBeenCalled();

    finishMigration(true);
    await applied;
    expect(AgentProfilesService.saveProfile).toHaveBeenCalledWith(
      "default",
      expect.objectContaining({ llm_profile_ref: "new-llm" }),
    );
  });

  it("creates `default` fresh when none is stored", async () => {
    vi.mocked(AgentProfilesService.getProfile)
      .mockRejectedValueOnce(new HttpError(404, "Not Found"))
      .mockResolvedValue({ name: "default", profile: STORED_DEFAULT } as never);

    await renderApply()({
      agent_kind: "openhands",
      llm_profile_ref: "new-llm",
    });

    expect(AgentProfilesService.saveProfile).toHaveBeenCalledWith("default", {
      agent_kind: "openhands",
      llm_profile_ref: "new-llm",
      public_skills: [...DEFAULT_ENABLED_SKILL_NAMES],
    });
  });

  it("leaves `default` untouched when it can't be read", async () => {
    vi.mocked(AgentProfilesService.getProfile).mockRejectedValue(
      new HttpError(500, "Internal Server Error"),
    );

    await renderApply()({
      agent_kind: "openhands",
      llm_profile_ref: "new-llm",
    });

    expect(AgentProfilesService.saveProfile).not.toHaveBeenCalled();
  });
});
