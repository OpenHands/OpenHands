import { QueryClient } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  launchConversation,
  resolveConversationLaunch,
} from "#/hooks/mutation/resolve-conversation-launch";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { SUBSCRIPTION_LOGIN_REQUIRED_ERROR } from "#/api/agent-server-adapter";

const {
  listAgentProfiles,
  getAgentProfile,
  listLlmProfiles,
  getLlmProfile,
  getOpenAIStatus,
  migrate,
} = vi.hoisted(() => ({
  listAgentProfiles: vi.fn(),
  getAgentProfile: vi.fn(),
  listLlmProfiles: vi.fn(),
  getLlmProfile: vi.fn(),
  getOpenAIStatus: vi.fn(),
  migrate: vi.fn(),
}));

vi.mock("#/hooks/use-migrate-default-agent-profile", () => ({
  ensureDefaultAgentProfileMigrated: migrate,
}));

vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  __esModule: true,
  default: { listProfiles: listAgentProfiles, getProfile: getAgentProfile },
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME: "default",
}));
vi.mock("#/api/profiles-service/profiles-service.api", () => ({
  __esModule: true,
  default: { listProfiles: listLlmProfiles, getProfile: getLlmProfile },
}));
vi.mock("#/api/llm-subscription-service", () => ({
  __esModule: true,
  default: { getOpenAIStatus },
}));

const REVIEWER = {
  id: "profile-1",
  name: "reviewer",
  agent_kind: "openhands",
  revision: 1,
  llm_profile_ref: "slow",
  mcp_server_refs: null,
};
const DEFAULT = {
  id: "profile-default",
  name: "default",
  agent_kind: "openhands",
  revision: 1,
  llm_profile_ref: "slow",
  mcp_server_refs: null,
};
const CLAUDE = {
  id: "profile-2",
  name: "claude",
  agent_kind: "acp",
  revision: 1,
  llm_profile_ref: null,
  mcp_server_refs: null,
};

function arrange({
  activeId = null as string | null,
  activeLlm = "slow" as string | null,
  detail = {} as Record<string, unknown>,
  llmProfiles = [{ name: "slow" }, { name: "fast" }],
} = {}) {
  listAgentProfiles.mockResolvedValue({
    profiles: [REVIEWER, CLAUDE, DEFAULT],
    active_agent_profile_id: activeId,
  });
  listLlmProfiles.mockResolvedValue({
    profiles: llmProfiles,
    active_profile: activeLlm,
  });
  migrate.mockResolvedValue(true);
  getAgentProfile.mockImplementation(async (name: string) => ({
    name,
    profile: {
      ...[REVIEWER, CLAUDE, DEFAULT].find((p) => p.name === name),
      ...detail,
    },
  }));
}

function launch(
  options: {
    agentProfileId?: string;
    llmProfile?: string;
    isCloud?: boolean;
  } = {},
) {
  return resolveConversationLaunch({
    queryClient: new QueryClient(),
    backendId: "local-1",
    orgId: null,
    isCloud: false,
    ...options,
  });
}

describe("resolveConversationLaunch", () => {
  afterEach(() => vi.resetAllMocks());

  it("runs the requested LLM on the picked profile", async () => {
    arrange();

    await expect(
      launch({ agentProfileId: "profile-1", llmProfile: "fast" }),
    ).resolves.toMatchObject({
      agentProfileId: "profile-1",
      llmProfileOverride: "fast",
      llmProfile: "fast",
    });
  });

  it("runs the active agent profile's own LLM on an implicit launch", async () => {
    arrange({ activeId: "profile-1", activeLlm: "fast" });

    const result = await launch();

    expect(result).toMatchObject({
      agentProfileId: "profile-1",
      llmProfile: "slow",
    });
    expect(result.llmProfileOverride).toBeUndefined();
  });

  it("runs default on the active LLM profile on an implicit launch", async () => {
    arrange({ activeId: "profile-default", activeLlm: "fast" });

    await expect(launch()).resolves.toMatchObject({
      llmProfileOverride: "fast",
      llmProfile: "fast",
    });
  });

  it("keeps an explicit pick on its own LLM", async () => {
    arrange({ activeLlm: "fast" });

    const result = await launch({ agentProfileId: "profile-1" });

    expect(result).toMatchObject({ llmProfile: "slow" });
    expect(result.llmProfileOverride).toBeUndefined();
  });

  it("runs default on the active LLM profile even when picked", async () => {
    arrange({ activeLlm: "fast" });

    await expect(
      launch({ agentProfileId: "profile-default" }),
    ).resolves.toMatchObject({
      llmProfileOverride: "fast",
      llmProfile: "fast",
    });
  });

  it("refuses a local launch when no profile is active", async () => {
    arrange();

    await expect(launch()).rejects.toThrow(
      "CONVERSATION$NO_ACTIVE_AGENT_PROFILE",
    );
  });

  it("fails a dangling-LLM-ref launch with the not-configured guidance", async () => {
    // The seeded `default` profile references the LLM profile "default",
    // which a never-configured account does not have (and has no active
    // LLM profile to fall back to) — the exact state from #18999.
    arrange({
      activeId: "profile-default",
      activeLlm: null,
      llmProfiles: [],
      detail: { llm_profile_ref: "default" },
    });

    await expect(launch({ agentProfileId: "profile-default" })).rejects.toThrow(
      "HOME$LLM_NOT_CONFIGURED_MESSAGE",
    );
  });

  it("still overrides a dangling ref when an LLM profile is requested", async () => {
    arrange({
      activeId: "profile-default",
      activeLlm: null,
      llmProfiles: [],
      detail: { llm_profile_ref: "default" },
    });

    await expect(
      launch({ agentProfileId: "profile-default", llmProfile: "fast" }),
    ).resolves.toMatchObject({ llmProfileOverride: "fast" });
  });

  it("leaves a dangling ref for the server when other LLM profiles exist", async () => {
    // Only a server with no LLM profile at all is "not configured"; a named
    // profile whose own LLM was deleted keeps its request so the server
    // reports the dangling ref.
    arrange({
      activeId: "profile-1",
      activeLlm: null,
      llmProfiles: [{ name: "fast" }],
    });

    await expect(
      launch({ agentProfileId: "profile-1" }),
    ).resolves.toMatchObject({
      agentProfileId: "profile-1",
      llmProfile: "slow",
    });
  });

  it("lets Cloud launch its own default when no profile is active", async () => {
    arrange();

    await expect(launch({ isCloud: true })).resolves.toEqual({});
    expect(getAgentProfile).not.toHaveBeenCalled();
  });

  it.each([
    [
      "enabled with a router",
      { enable_classify_and_switch_llm_tool: true, meta_profile_ref: "router" },
      true,
    ],
    [
      "enabled without a name, so the first router",
      { enable_classify_and_switch_llm_tool: true, meta_profile_ref: null },
      true,
    ],
    [
      "disabled",
      {
        enable_classify_and_switch_llm_tool: false,
        meta_profile_ref: "router",
      },
      false,
    ],
  ])(
    "reports the profile's router when the routing tool is %s",
    async (_case, detail, expected) => {
      arrange({ activeId: "profile-1", detail });

      await expect(launch()).resolves.toMatchObject({
        profileHasRouter: expected,
      });
    },
  );

  it("reports the ACP profile's server for the conversation tag", async () => {
    arrange({ activeId: "profile-2", detail: { acp_server: "claude-code" } });

    await expect(launch()).resolves.toMatchObject({
      agentProfileId: "profile-2",
      agentProfileKind: "acp",
      acpServer: "claude-code",
      profileHasRouter: false,
    });
  });

  it("reads the profiles only after the default-profile migration settles", async () => {
    arrange({ activeId: "profile-1" });
    let finishMigration = () => {};
    migrate.mockReturnValue(
      new Promise<true>((resolve) => {
        finishMigration = () => resolve(true);
      }),
    );

    const pending = launch();
    await Promise.resolve();
    expect(listAgentProfiles).not.toHaveBeenCalled();

    finishMigration();
    await expect(pending).resolves.toMatchObject({
      agentProfileId: "profile-1",
    });
  });

  it("fails a local launch when the migration fails, and skips it on Cloud", async () => {
    arrange({ activeId: "profile-1" });
    migrate.mockRejectedValue(new Error("offline"));

    await expect(launch()).rejects.toThrow("offline");
    expect(listAgentProfiles).not.toHaveBeenCalled();

    migrate.mockClear();
    await launch({ isCloud: true });
    expect(migrate).not.toHaveBeenCalled();
  });

  it("still launches the profile when its detail cannot be read", async () => {
    arrange({ activeId: "profile-1" });
    getAgentProfile.mockRejectedValue(new Error("boom"));

    await expect(launch()).resolves.toMatchObject({
      agentProfileId: "profile-1",
      profileHasRouter: false,
    });
  });

  it("launches Cloud's own default when the profile's LLM profile is gone", async () => {
    arrange({ activeId: "profile-1" });
    listLlmProfiles.mockResolvedValue({
      profiles: [{ name: "fast" }],
      active_profile: "fast",
    });

    await expect(launch({ isCloud: true })).resolves.toEqual({});
  });

  it("keeps a secret-restricted profile on Cloud even when its LLM profile is gone", async () => {
    arrange({ activeId: "profile-1", detail: { secret_refs: [] } });
    listLlmProfiles.mockResolvedValue({
      profiles: [{ name: "fast" }],
      active_profile: "fast",
    });

    await expect(launch({ isCloud: true })).resolves.toMatchObject({
      agentProfileId: "profile-1",
    });
  });

  it("launches the profile on Cloud when its LLM profile exists", async () => {
    arrange({ activeId: "profile-1" });

    await expect(launch({ isCloud: true })).resolves.toEqual({
      agentProfileId: "profile-1",
      agentProfileKind: "openhands",
      llmProfile: "slow",
    });
    expect(getAgentProfile).not.toHaveBeenCalled();
  });
});

describe("launchConversation", () => {
  afterEach(() => vi.restoreAllMocks());

  it("refuses to launch on a subscription LLM that isn't connected", async () => {
    arrange({ activeId: "profile-1", activeLlm: "slow" });
    getLlmProfile.mockResolvedValue({
      name: "slow",
      config: { model: "openai/gpt-5", auth_type: "subscription" },
    });
    getOpenAIStatus.mockResolvedValue({ connected: false });
    const create = vi.spyOn(
      AgentServerConversationService,
      "createConversation",
    );

    await expect(
      launchConversation(
        {
          queryClient: new QueryClient(),
          backendId: "local-1",
          orgId: null,
          isCloud: false,
        },
        {},
      ),
    ).rejects.toThrow(SUBSCRIPTION_LOGIN_REQUIRED_ERROR);
    expect(getLlmProfile).toHaveBeenCalledWith("slow");
    expect(create).not.toHaveBeenCalled();
  });
});
