import { beforeEach, describe, expect, it, vi } from "vitest";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import {
  applyPreferredHostLaunch,
  CLAUDE_CODE_AUTO_AGENT_PROFILE_NAME,
} from "#/hooks/use-ensure-preferred-subscription-launch";

vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  default: {
    getProfile: vi.fn(),
    saveProfile: vi.fn(),
  },
}));

describe("applyPreferredHostLaunch", () => {
  const activateLlm = vi.fn();
  const activateAgent = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("activates a ChatGPT auto LLM profile", async () => {
    await applyPreferredHostLaunch({
      launch: { kind: "chatgpt" },
      llmProfiles: {
        active_profile: "default",
        profiles: [
          {
            name: "default",
            model: "gpt-5.5",
            base_url: null,
            api_key_set: false,
          },
          {
            name: "sub-chatgpt-gpt-5-2",
            model: "gpt-5.2",
            base_url: null,
            api_key_set: false,
          },
        ],
      },
      agentProfiles: { profiles: [], active_agent_profile_id: null },
      activateLlm,
      activateAgent,
    });

    expect(activateLlm).toHaveBeenCalledWith("sub-chatgpt-gpt-5-2");
    expect(activateAgent).not.toHaveBeenCalled();
  });

  it("activates an existing Claude Code agent profile", async () => {
    await applyPreferredHostLaunch({
      launch: {
        kind: "acp",
        acpServer: "claude-code",
        profileName: CLAUDE_CODE_AUTO_AGENT_PROFILE_NAME,
      },
      llmProfiles: { active_profile: null, profiles: [] },
      agentProfiles: {
        active_agent_profile_id: "openhands-id",
        profiles: [
          {
            id: "openhands-id",
            name: "default",
            agent_kind: "openhands",
            revision: 1,
            llm_profile_ref: "default",
            mcp_server_refs: null,
          },
          {
            id: "claude-id",
            name: CLAUDE_CODE_AUTO_AGENT_PROFILE_NAME,
            agent_kind: "acp",
            revision: 1,
            llm_profile_ref: null,
            mcp_server_refs: null,
          },
        ],
      },
      activateLlm,
      activateAgent,
    });

    expect(activateAgent).toHaveBeenCalledWith("claude-id");
    expect(AgentProfilesService.saveProfile).not.toHaveBeenCalled();
  });

  it("creates and activates a Claude Code agent profile when none exists", async () => {
    vi.mocked(AgentProfilesService.saveProfile).mockResolvedValue({
      name: CLAUDE_CODE_AUTO_AGENT_PROFILE_NAME,
    } as Awaited<ReturnType<typeof AgentProfilesService.saveProfile>>);
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      profile: {
        id: "new-claude",
        name: CLAUDE_CODE_AUTO_AGENT_PROFILE_NAME,
        agent_kind: "acp",
        acp_server: "claude-code",
        revision: 1,
        mcp_server_refs: null,
        acp_model: "default",
        acp_session_mode: null,
        acp_prompt_timeout: 0,
        acp_command: null,
        acp_args: null,
      },
    } as Awaited<ReturnType<typeof AgentProfilesService.getProfile>>);

    await applyPreferredHostLaunch({
      launch: {
        kind: "acp",
        acpServer: "claude-code",
        profileName: CLAUDE_CODE_AUTO_AGENT_PROFILE_NAME,
      },
      llmProfiles: { active_profile: null, profiles: [] },
      agentProfiles: {
        active_agent_profile_id: null,
        profiles: [],
      },
      activateLlm,
      activateAgent,
    });

    expect(AgentProfilesService.saveProfile).toHaveBeenCalledWith(
      CLAUDE_CODE_AUTO_AGENT_PROFILE_NAME,
      expect.objectContaining({
        agent_kind: "acp",
        acp_server: "claude-code",
      }),
    );
    expect(activateAgent).toHaveBeenCalledWith("new-claude");
  });

  it("creates and activates a Cursor CLI custom ACP profile", async () => {
    vi.mocked(AgentProfilesService.saveProfile).mockResolvedValue({
      name: "cursor-cli",
    } as Awaited<ReturnType<typeof AgentProfilesService.saveProfile>>);
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      profile: {
        id: "new-cursor",
        name: "cursor-cli",
        agent_kind: "acp",
        acp_server: "custom",
        revision: 1,
        mcp_server_refs: null,
        acp_model: null,
        acp_session_mode: null,
        acp_prompt_timeout: 0,
        acp_command: "agent acp",
        acp_args: null,
      },
    } as Awaited<ReturnType<typeof AgentProfilesService.getProfile>>);

    await applyPreferredHostLaunch({
      launch: {
        kind: "acp",
        acpServer: "custom",
        profileName: "cursor-cli",
        command: ["agent", "acp"],
      },
      llmProfiles: { active_profile: null, profiles: [] },
      agentProfiles: { active_agent_profile_id: null, profiles: [] },
      activateLlm,
      activateAgent,
    });

    expect(AgentProfilesService.saveProfile).toHaveBeenCalledWith(
      "cursor-cli",
      expect.objectContaining({
        agent_kind: "acp",
        acp_server: "custom",
        acp_command: "agent acp",
      }),
    );
    expect(activateAgent).toHaveBeenCalledWith("new-cursor");
  });
});
