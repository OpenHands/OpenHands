import { beforeEach, describe, expect, it, vi } from "vitest";
import { CANVAS_UI_CLIENT_TOOL_NAME } from "#/constants/canvas-ui";
import { LAUNCH_CHILD_CONVERSATION_TOOL_NAME } from "#/constants/child-conversation";
import { DEFAULT_SETTINGS } from "#/services/settings";
import type { Settings } from "#/types/settings";
import {
  AGENT_CANVAS_SOURCE,
  CLIENT_SOURCE_TAG_KEY,
  buildStartConversationRequest,
  buildStartPlanningConversationRequest,
  buildStartPlanningConversationRequestWithEncryptedSettings,
  toConversationPage,
} from "./agent-server-adapter";
import SettingsService from "./settings-service/settings-service.api";
import AgentProfilesService from "./agent-profiles-service/agent-profiles-service.api";
import ProfilesService from "./profiles-service/profiles-service.api";

vi.mock("./settings-service/settings-service.api", () => ({
  default: { getSettingsForConversation: vi.fn() },
}));
vi.mock("./secrets-service", () => ({
  SecretsService: { getSecrets: vi.fn().mockResolvedValue([]) },
}));
vi.mock("./agent-profiles-service/agent-profiles-service.api", () => ({
  default: { listProfiles: vi.fn() },
}));
vi.mock("./profiles-service/profiles-service.api", () => ({
  default: { getProfile: vi.fn() },
}));

const encryptedValue = "gAAAAAencrypted-mcp-header";

function makeSettings(agentSettings: Settings["agent_settings"]): Settings {
  return {
    ...DEFAULT_SETTINGS,
    agent_settings: agentSettings,
    conversation_settings: {
      confirmation_mode: false,
      security_analyzer: null,
      max_iterations: 20,
    },
  };
}

function getAgentContextSkillNames(
  payload: ReturnType<typeof buildStartConversationRequest>,
): Array<string | undefined> {
  const agentSettings = payload.agent_settings as
    | {
        agent_context?: {
          skills?: Array<{ name?: string }>;
        };
      }
    | undefined;
  return agentSettings?.agent_context?.skills?.map((skill) => skill.name) ?? [];
}

function getPlannerAgentContextSkillNames(payload: {
  agent: { agent_context?: { skills?: Array<{ name?: string }> } };
}): Array<string | undefined> {
  return payload.agent.agent_context?.skills?.map((skill) => skill.name) ?? [];
}

describe("buildStartConversationRequest", () => {
  it("marks OpenHands start requests as encrypted when MCP headers are encrypted", () => {
    const agentSettings = {
      agent_kind: "openhands",
      llm: {
        model: "litellm_proxy/openai/gpt-5.5",
        api_key: "gAAAAAencrypted-llm-api-key",
      },
      mcp_config: {
        linear: {
          url: "https://mcp.linear.app/mcp",
          transport: "http",
          headers: {
            Authorization: encryptedValue,
          },
        },
      },
    };
    const settings = makeSettings(agentSettings);

    const payload = buildStartConversationRequest({
      settings,
      encryptedAgentSettings: agentSettings,
      encryptedConversationSettings: settings.conversation_settings!,
      secretsEncrypted: true,
    });

    expect(payload.agent_settings!.agent_kind).toBe("openhands");
    expect(payload.agent_settings!.mcp_config).toEqual(
      agentSettings.mcp_config,
    );
    expect(payload.secrets_encrypted).toBe(true);
  });

  it("marks ACP start requests as encrypted when MCP headers are encrypted", () => {
    const agentSettings = {
      agent_kind: "acp",
      acp_server: "codex",
      acp_command: ["codex-acp"],
      acp_model: "gpt-5.5/medium",
      mcp_config: {
        linear: {
          url: "https://mcp.linear.app/mcp",
          transport: "http",
          headers: {
            Authorization: encryptedValue,
          },
        },
      },
    };
    const settings = makeSettings(agentSettings);

    const payload = buildStartConversationRequest({
      settings,
      encryptedAgentSettings: agentSettings,
      encryptedConversationSettings: settings.conversation_settings!,
      secretsEncrypted: true,
    });

    expect(payload.agent_settings!.agent_kind).toBe("acp");
    expect(payload.agent_settings!.mcp_config).toEqual(
      agentSettings.mcp_config,
    );
    expect(payload.secrets_encrypted).toBe(true);
  });

  it("builds a raw planning agent request for local Planner", () => {
    const agentSettings = {
      agent_kind: "openhands",
      llm: {
        model: "openhands/minimax-m2.7",
        api_key: "gAAAAAencrypted-llm-api-key",
      },
    };

    const payload = buildStartPlanningConversationRequest({
      encryptedAgentSettings: agentSettings,
      workingDir: "/workspace/project/agent-canvas",
      parentConversationId: "parent-1",
      secretsEncrypted: true,
      customSecrets: [{ name: "CUSTOM_TOKEN" }],
    });

    expect(payload.agent).toMatchObject({
      kind: "Agent",
      system_prompt_filename: "system_prompt_planning.j2",
      system_prompt_kwargs: {
        plan_structure: expect.stringContaining("OBJECTIVE"),
      },
      llm: {
        model: "openhands/minimax-m2.7",
        api_key: "gAAAAAencrypted-llm-api-key",
      },
      tools: [
        { name: "glob", params: {} },
        { name: "grep", params: {} },
        {
          name: "planning_file_editor",
          params: {
            plan_path: "/workspace/project/agent-canvas/.agents_tmp/PLAN.md",
          },
        },
      ],
      // Matches the SDK planning preset's get_planning_condenser.
      condenser: {
        kind: "LLMSummarizingCondenser",
        max_size: 100,
        keep_first: 6,
        llm: {
          model: "openhands/minimax-m2.7",
          usage_id: "planning_condenser",
        },
      },
    });
    expect(payload.agent_settings).toBeUndefined();
    expect(payload.worktree).toBe(false);
    expect(payload.tags).toEqual({ plannerparent: "parent-1" });
    expect(payload.secrets_encrypted).toBe(true);
    expect(payload.secrets).toHaveProperty("CUSTOM_TOKEN");
  });

  it("links the local planner to its parent server-side so the relationship survives storage loss", () => {
    const payload = buildStartPlanningConversationRequest({
      encryptedAgentSettings: {
        agent_kind: "openhands",
        llm: { model: "openhands/minimax-m2.7" },
      },
      workingDir: "/workspace/project",
      parentConversationId: "parent-1",
    });

    // The agent-server derives the parent's `sub_conversation_ids` from this,
    // which is what makes the planner recoverable without localStorage.
    expect(payload.parent_conversation_id).toBe("parent-1");
  });

  it("defaults the planner's max_iterations to 500 when not provided", () => {
    const payload = buildStartPlanningConversationRequest({
      encryptedAgentSettings: {
        agent_kind: "openhands",
        llm: { model: "openhands/minimax-m2.7" },
      },
      workingDir: "/workspace/project",
      parentConversationId: "parent-1",
    });

    expect(payload.max_iterations).toBe(500);
  });

  it("uses the caller-provided max_iterations instead of the hardcoded default", () => {
    const payload = buildStartPlanningConversationRequest({
      encryptedAgentSettings: {
        agent_kind: "openhands",
        llm: { model: "openhands/minimax-m2.7" },
      },
      workingDir: "/workspace/project",
      parentConversationId: "parent-1",
      maxIterations: 250,
    });

    expect(payload.max_iterations).toBe(250);
  });

  it("omits local planner helper conversations from paginated conversation results", () => {
    const page = toConversationPage({
      items: [
        {
          id: "main-1",
          created_at: "2024-01-01T00:00:00.000Z",
          updated_at: "2024-01-01T00:00:00.000Z",
          execution_status: "idle",
          tags: {},
        },
        {
          id: "plan-1",
          created_at: "2024-01-01T00:00:00.000Z",
          updated_at: "2024-01-01T00:00:00.000Z",
          execution_status: "idle",
          tags: { plannerparent: "main-1" },
        },
      ],
    });

    expect(page.items.map((item) => item.id)).toEqual(["main-1"]);
  });

  it("keeps ACP start requests unencrypted when no encrypted MCP values are present", () => {
    const agentSettings = {
      agent_kind: "acp",
      acp_server: "codex",
      acp_command: ["codex-acp"],
      acp_model: "gpt-5.5/medium",
      mcp_config: {
        publicDocs: {
          url: "https://docs.example.com/mcp",
          transport: "http",
        },
      },
    };
    const settings = makeSettings(agentSettings);

    const payload = buildStartConversationRequest({
      settings,
      encryptedAgentSettings: agentSettings,
      encryptedConversationSettings: settings.conversation_settings!,
      secretsEncrypted: true,
    });

    expect(payload.agent_settings!.agent_kind).toBe("acp");
    expect(payload.secrets_encrypted).toBeUndefined();
  });

  it("ships only allow-listed catalog skills to a OpenHands conversation context", () => {
    const settings = makeSettings({
      agent_kind: "openhands",
      llm: {
        model: "litellm_proxy/openai/gpt-5.5",
        api_key: "sk-test",
      },
      agent_context: {
        skills: [
          { name: "disabled-custom", content: "disabled" },
          { name: "enabled-custom", content: "enabled" },
        ],
      },
    });
    settings.disabled_skills = ["agent-memory", "disabled-custom"];

    const payload = buildStartConversationRequest({ settings });
    const skillNames = getAgentContextSkillNames(payload);

    // `agent-memory` is default-enabled, so this asserts the deny-list still
    // wins over the allow-list — the case that matters before the one-shot
    // migration has run.
    expect(skillNames).not.toContain("agent-memory");
    expect(skillNames).not.toContain("disabled-custom");
    // Skills already on the agent context are user-authored and stay opt-out.
    expect(skillNames).toContain("enabled-custom");
    // No `enabled_skills` on the settings means the curated default applies:
    // `add-skill` is flagged `defaultEnabled` in the catalog, `add-javadoc` is
    // not, and shipping every catalog skill is what OpenHands#16302 reported.
    expect(skillNames).toContain("add-skill");
    expect(skillNames).not.toContain("add-javadoc");

    expect(payload.agent_settings?.agent_context?.disabled_skills).toEqual([
      "agent-memory",
      "disabled-custom",
    ]);
  });

  it("ships only allow-listed catalog skills to a ACP conversation context", () => {
    const settings = makeSettings({
      agent_kind: "acp",
      acp_server: "codex",
      acp_command: ["codex-acp"],
      acp_model: "gpt-5.5/medium",
      agent_context: {
        skills: [
          { name: "disabled-custom", content: "disabled" },
          { name: "enabled-custom", content: "enabled" },
        ],
      },
    });
    settings.disabled_skills = ["agent-memory", "disabled-custom"];

    const payload = buildStartConversationRequest({ settings });
    const skillNames = getAgentContextSkillNames(payload);

    // `agent-memory` is default-enabled, so this asserts the deny-list still
    // wins over the allow-list — the case that matters before the one-shot
    // migration has run.
    expect(skillNames).not.toContain("agent-memory");
    expect(skillNames).not.toContain("disabled-custom");
    // Skills already on the agent context are user-authored and stay opt-out.
    expect(skillNames).toContain("enabled-custom");
    // No `enabled_skills` on the settings means the curated default applies:
    // `add-skill` is flagged `defaultEnabled` in the catalog, `add-javadoc` is
    // not, and shipping every catalog skill is what OpenHands#16302 reported.
    expect(skillNames).toContain("add-skill");
    expect(skillNames).not.toContain("add-javadoc");

    expect(payload.agent_settings?.agent_context?.disabled_skills).toEqual([
      "agent-memory",
      "disabled-custom",
    ]);
  });

  it("loads a catalog skill the opening slash command invokes", () => {
    // An automation card fills the chat input with the skill's own command
    // (`findAutomationCommand`), and 18 of the catalog's 24 slash commands
    // belong to skills that are off by default — without this the card would
    // silently do nothing.
    const settings = makeSettings({
      agent_kind: "openhands",
      llm: { model: "litellm_proxy/openai/gpt-5.5", api_key: "sk-test" },
    });

    const payload = buildStartConversationRequest({
      settings,
      query: "/standup-digest:setup",
    });

    expect(getAgentContextSkillNames(payload)).toContain(
      "slack-standup-digest",
    );
  });

  it("does not admit a skill named mid-sentence rather than invoked", () => {
    const settings = makeSettings({
      agent_kind: "openhands",
      llm: { model: "litellm_proxy/openai/gpt-5.5", api_key: "sk-test" },
    });

    const payload = buildStartConversationRequest({
      settings,
      query: "tell me what /standup-digest:setup would do",
    });

    expect(getAgentContextSkillNames(payload)).not.toContain(
      "slack-standup-digest",
    );
  });

  it("lets an invoked skill override a stored deny entry for that conversation", () => {
    const settings = makeSettings({
      agent_kind: "openhands",
      llm: { model: "litellm_proxy/openai/gpt-5.5", api_key: "sk-test" },
    });
    settings.disabled_skills = ["slack-standup-digest"];

    const payload = buildStartConversationRequest({
      settings,
      query: "/standup-digest:setup",
    });

    expect(getAgentContextSkillNames(payload)).toContain(
      "slack-standup-digest",
    );
  });
});

describe("buildStartConversationRequest — execution runtime", () => {
  it("requests DockerExecutionWorkspace from Docker execution servers", () => {
    const payload = buildStartConversationRequest({
      settings: makeSettings({ agent_kind: "openhands" }),
      workingDir: "/host/workspace",
      executionRuntime: "docker",
    });

    expect(payload.workspace).toEqual({
      kind: "DockerExecutionWorkspace",
      working_dir: "/workspace",
    });
    expect(payload.worktree).toBe(false);
  });

  it("keeps LocalWorkspace for local and older servers", () => {
    const payload = buildStartConversationRequest({
      settings: makeSettings({ agent_kind: "openhands" }),
      workingDir: "/host/workspace",
    });

    expect(payload.workspace).toEqual({
      kind: "LocalWorkspace",
      working_dir: "/host/workspace",
    });
  });
});

describe("buildStartPlanningConversationRequestWithEncryptedSettings", () => {
  const globallyActiveSettings = {
    agentSettings: {
      agent_kind: "openhands",
      llm: {
        model: "openhands/globally-active-model",
        api_key: "gAAAAAglobal-key",
        base_url: "https://global.example.com",
      },
    },
    conversationSettings: {},
    secretsEncrypted: true,
    skillEnablement: { disabledSkills: [] },
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(SettingsService.getSettingsForConversation).mockResolvedValue(
      globallyActiveSettings,
    );
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: [
        {
          id: "profile-parent",
          name: "parent-profile",
          agent_kind: "openhands",
          revision: 1,
          llm_profile_ref: "parent-llm",
          mcp_server_refs: null,
        },
      ],
      active_agent_profile_id: "profile-other",
    });
    vi.mocked(ProfilesService.getProfile).mockResolvedValue({
      name: "parent-llm",
      api_key_set: true,
      config: {
        model: "openhands/parent-profile-model",
        api_key: "gAAAAAparent-key",
        base_url: "https://parent.example.com",
      },
    });
  });

  it("pins the planner to the parent's launched profile, not the globally active one", async () => {
    const payload =
      await buildStartPlanningConversationRequestWithEncryptedSettings({
        workingDir: "/workspace/project",
        parentConversationId: "parent-1",
        parentAgentProfileId: "profile-parent",
      });

    expect(ProfilesService.getProfile).toHaveBeenCalledWith(
      "parent-llm",
      "encrypted",
    );
    expect(payload.agent.llm).toMatchObject({
      model: "openhands/parent-profile-model",
      api_key: "gAAAAAparent-key",
      base_url: "https://parent.example.com",
    });
  });

  it("falls back to global settings when the parent was not launched from a profile", async () => {
    const payload =
      await buildStartPlanningConversationRequestWithEncryptedSettings({
        workingDir: "/workspace/project",
        parentConversationId: "parent-1",
        parentAgentProfileId: null,
      });

    expect(AgentProfilesService.listProfiles).not.toHaveBeenCalled();
    expect(payload.agent.llm).toMatchObject({
      model: "openhands/globally-active-model",
    });
  });

  it("prefers the parent conversation's current active_profile over its launched agent profile", async () => {
    // The conversation launched under "parent-profile" (-> "parent-llm"), but
    // was later switched to a different LLM profile via /model or the
    // agent's own SwitchLLMTool — active_profile tracks that switch.
    vi.mocked(ProfilesService.getProfile).mockImplementation(async (name) =>
      name === "switched-llm"
        ? {
            name: "switched-llm",
            api_key_set: true,
            config: {
              model: "openhands/switched-model",
              api_key: "gAAAAAswitched-key",
              base_url: "https://switched.example.com",
            },
          }
        : {
            name: "parent-llm",
            api_key_set: true,
            config: {
              model: "openhands/parent-profile-model",
              api_key: "gAAAAAparent-key",
              base_url: "https://parent.example.com",
            },
          },
    );

    const payload =
      await buildStartPlanningConversationRequestWithEncryptedSettings({
        workingDir: "/workspace/project",
        parentConversationId: "parent-1",
        parentActiveProfileName: "switched-llm",
        parentAgentProfileId: "profile-parent",
      });

    expect(ProfilesService.getProfile).toHaveBeenCalledWith(
      "switched-llm",
      "encrypted",
    );
    // The launched agent profile must not even be consulted once
    // active_profile resolves successfully.
    expect(AgentProfilesService.listProfiles).not.toHaveBeenCalled();
    expect(payload.agent.llm).toMatchObject({
      model: "openhands/switched-model",
      api_key: "gAAAAAswitched-key",
      base_url: "https://switched.example.com",
    });
  });

  it("falls back to the launched agent profile when active_profile can't be resolved", async () => {
    vi.mocked(ProfilesService.getProfile).mockImplementation(async (name) => {
      if (name === "dangling-llm") throw new Error("not found");
      return {
        name: "parent-llm",
        api_key_set: true,
        config: {
          model: "openhands/parent-profile-model",
          api_key: "gAAAAAparent-key",
          base_url: "https://parent.example.com",
        },
      };
    });

    const payload =
      await buildStartPlanningConversationRequestWithEncryptedSettings({
        workingDir: "/workspace/project",
        parentConversationId: "parent-1",
        parentActiveProfileName: "dangling-llm",
        parentAgentProfileId: "profile-parent",
      });

    expect(payload.agent.llm).toMatchObject({
      model: "openhands/parent-profile-model",
    });
  });

  it("falls back to global settings when the parent's profile reference dangles", async () => {
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: [],
      active_agent_profile_id: null,
    });

    const payload =
      await buildStartPlanningConversationRequestWithEncryptedSettings({
        workingDir: "/workspace/project",
        parentConversationId: "parent-1",
        parentAgentProfileId: "profile-parent",
      });

    expect(ProfilesService.getProfile).not.toHaveBeenCalled();
    expect(payload.agent.llm).toMatchObject({
      model: "openhands/globally-active-model",
    });
  });

  it("mirrors the parent's configured max_iterations instead of hardcoding a lower cap", async () => {
    vi.mocked(SettingsService.getSettingsForConversation).mockResolvedValue({
      ...globallyActiveSettings,
      conversationSettings: { max_iterations: 1000 },
    });

    const payload =
      await buildStartPlanningConversationRequestWithEncryptedSettings({
        workingDir: "/workspace/project",
        parentConversationId: "parent-1",
      });

    expect(payload.max_iterations).toBe(1000);
  });

  it("falls back to 500 when the parent has no configured max_iterations", async () => {
    const payload =
      await buildStartPlanningConversationRequestWithEncryptedSettings({
        workingDir: "/workspace/project",
        parentConversationId: "parent-1",
      });

    expect(payload.max_iterations).toBe(500);
  });

  it("applies the full skill enablement to the planner's agent context, mirroring the code agent", async () => {
    // Both lists, because they cover different populations (#16302): the
    // allow-list gates the bundled catalog, the deny-list still wins over it.
    // `add-javadoc` is a catalog skill outside the recommended defaults, so
    // it only loads because the allow-list names it.
    vi.mocked(SettingsService.getSettingsForConversation).mockResolvedValue({
      ...globallyActiveSettings,
      skillEnablement: {
        enabledSkills: ["add-javadoc", "agent-memory"],
        disabledSkills: ["agent-memory"],
      },
    });

    const payload =
      await buildStartPlanningConversationRequestWithEncryptedSettings({
        workingDir: "/workspace/project",
        parentConversationId: "parent-1",
      });

    const skillNames = getPlannerAgentContextSkillNames(payload);
    expect(skillNames).not.toContain("agent-memory");
    expect(skillNames).toContain("add-javadoc");
  });
});

describe("buildStartConversationRequest — agentProfileId path", () => {
  it("sends agent_profile_id and omits agent_settings (mutually exclusive)", () => {
    const settings = makeSettings({
      agent_kind: "openhands",
      llm: { model: "litellm_proxy/openai/gpt-5.5", api_key: "sk-test" },
    });

    const payload = buildStartConversationRequest({
      settings,
      agentProfileId: "profile-xyz",
      agentProfileKind: "openhands",
    });

    expect(payload.agent_profile_id).toBe("profile-xyz");
    expect(payload.agent_settings).toBeUndefined();
    expect(payload.client_tools.map((tool) => tool.name)).toEqual([
      CANVAS_UI_CLIENT_TOOL_NAME,
      LAUNCH_CHILD_CONVERSATION_TOOL_NAME,
    ]);
  });

  it("suppresses the ACP server tag when launching from a profile", () => {
    const agentSettings = {
      agent_kind: "acp",
      acp_server: "codex",
      acp_command: ["codex-acp"],
      acp_model: "gpt-5.5/medium",
    };

    // Without a profile the ACP server tag is stamped from settings...
    expect(
      buildStartConversationRequest({ settings: makeSettings(agentSettings) })
        .tags,
    ).toBeDefined();

    // ...but a profile launch resolves the server server-side, so the tag
    // (which may not match the launched profile) is omitted while the client
    // source telemetry tag is still stamped.
    const payload = buildStartConversationRequest({
      settings: makeSettings(agentSettings),
      agentProfileId: "profile-xyz",
    });
    expect(payload.tags).toEqual({
      [CLIENT_SOURCE_TAG_KEY]: AGENT_CANVAS_SOURCE,
    });
  });

  it("suppresses secrets_encrypted when launching from a profile", () => {
    const agentSettings = {
      agent_kind: "openhands",
      llm: {
        model: "litellm_proxy/openai/gpt-5.5",
        api_key: "gAAAAAencrypted-llm-api-key",
      },
      mcp_config: {
        mcpServers: {
          linear: {
            url: "https://mcp.linear.app/mcp",
            transport: "http",
            headers: { Authorization: encryptedValue },
          },
        },
      },
    };
    const settings = makeSettings(agentSettings);

    // Same inputs without a profile would set secrets_encrypted (covered
    // above); the profile path defers secret resolution to the server.
    const payload = buildStartConversationRequest({
      settings,
      encryptedAgentSettings: agentSettings,
      encryptedConversationSettings: settings.conversation_settings!,
      secretsEncrypted: true,
      agentProfileId: "profile-xyz",
    });

    expect(payload.secrets_encrypted).toBeUndefined();
  });
});
