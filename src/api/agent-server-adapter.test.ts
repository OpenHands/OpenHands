import { describe, expect, it } from "vitest";
import { CANVAS_UI_CLIENT_TOOL_NAME } from "#/constants/canvas-ui";
import { DEFAULT_SETTINGS } from "#/services/settings";
import type { Settings } from "#/types/settings";
import {
  buildStartConversationRequest,
  buildStartPlanningConversationRequest,
  toConversationPage,
} from "./agent-server-adapter";

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
    // (which may not match the launched profile) is omitted.
    const payload = buildStartConversationRequest({
      settings: makeSettings(agentSettings),
      agentProfileId: "profile-xyz",
    });
    expect(payload.tags).toBeUndefined();
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
