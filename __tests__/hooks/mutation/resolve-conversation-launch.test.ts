import { QueryClient } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import { resolveConversationLaunch } from "#/hooks/mutation/resolve-conversation-launch";

const { listAgentProfiles, getAgentProfile, listLlmProfiles } = vi.hoisted(
  () => ({
    listAgentProfiles: vi.fn(),
    getAgentProfile: vi.fn(),
    listLlmProfiles: vi.fn(),
  }),
);

vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  __esModule: true,
  default: { listProfiles: listAgentProfiles, getProfile: getAgentProfile },
}));
vi.mock("#/api/profiles-service/profiles-service.api", () => ({
  __esModule: true,
  default: { listProfiles: listLlmProfiles },
}));

const REVIEWER = {
  id: "profile-1",
  name: "reviewer",
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
} = {}) {
  listAgentProfiles.mockResolvedValue({
    profiles: [REVIEWER, CLAUDE],
    active_agent_profile_id: activeId,
  });
  listLlmProfiles.mockResolvedValue({
    profiles: [{ name: "slow" }, { name: "fast" }],
    active_profile: activeLlm,
  });
  getAgentProfile.mockImplementation(async (name: string) => ({
    name,
    profile: {
      ...[REVIEWER, CLAUDE].find((p) => p.name === name),
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

  it("runs the active LLM profile on an implicit launch", async () => {
    arrange({ activeId: "profile-1", activeLlm: "fast" });

    await expect(launch()).resolves.toMatchObject({
      agentProfileId: "profile-1",
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

  it("refuses a local launch when no profile is active", async () => {
    arrange();

    await expect(launch()).rejects.toThrow(
      "CONVERSATION$NO_ACTIVE_AGENT_PROFILE",
    );
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
      "enabled without a router",
      { enable_classify_and_switch_llm_tool: true, meta_profile_ref: null },
      false,
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

  it("still launches the profile when its detail cannot be read", async () => {
    arrange({ activeId: "profile-1" });
    getAgentProfile.mockRejectedValue(new Error("boom"));

    await expect(launch()).resolves.toMatchObject({
      agentProfileId: "profile-1",
      profileHasRouter: false,
    });
  });
});
