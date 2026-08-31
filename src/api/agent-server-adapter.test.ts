import { describe, expect, it } from "vitest";
import { CANVAS_UI_CLIENT_TOOL_NAME } from "#/constants/canvas-ui";
import { LAUNCH_CHILD_CONVERSATION_TOOL_NAME } from "#/constants/child-conversation";
import { DEFAULT_SETTINGS } from "#/services/settings";
import type { Settings } from "#/types/settings";
import {
  AGENT_CANVAS_SOURCE,
  CLIENT_SOURCE_TAG_KEY,
  buildStartConversationRequest,
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
