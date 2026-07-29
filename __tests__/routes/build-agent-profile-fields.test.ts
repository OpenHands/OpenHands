import { describe, expect, it } from "vitest";
import { buildAgentProfileFields } from "#/routes/agent-settings";
import type { SettingsFieldSchema } from "#/types/settings";

const baseAcp = {
  isAcp: true,
  selectedPreset: "claude-code",
  isDefaultProviderCommand: true,
  commandTokens: ["npx", "-y", "@zed-industries/claude-code-acp"],
  acpModel: "claude-opus-4-8",
  subAgentsEnabled: false,
  toolConcurrencyField: undefined,
  toolConcurrency: "",
};

const concurrencyField: SettingsFieldSchema = {
  key: "tool_concurrency_limit",
  label: "Tool concurrency limit",
  section: "agent",
  section_label: "Agent",
  value_type: "integer",
  choices: [],
  depends_on: [],
  prominence: "minor",
  secret: false,
  required: false,
};

describe("buildAgentProfileFields — ACP", () => {
  it("stores no explicit command for a built-in provider on its default command", () => {
    const fields = buildAgentProfileFields(baseAcp);
    expect(fields).toEqual({
      agent_kind: "acp",
      acp_server: "claude-code",
      acp_model: "claude-opus-4-8",
      acp_command: null,
      acp_args: null,
    });
  });

  it("stores the verbatim shell command when it diverges from the default", () => {
    const fields = buildAgentProfileFields({
      ...baseAcp,
      isDefaultProviderCommand: false,
      commandTokens: ["npx", "-y", "@zed-industries/claude-code-acp@0.5.0"],
    });
    expect(fields.agent_kind).toBe("acp");
    if (fields.agent_kind === "acp") {
      expect(fields.acp_command).toBe(
        "npx -y @zed-industries/claude-code-acp@0.5.0",
      );
    }
  });

  it("stores the command for the custom preset even if it happens to match a default", () => {
    const fields = buildAgentProfileFields({
      ...baseAcp,
      selectedPreset: "custom",
      // A custom preset is never treated as a built-in default.
      isDefaultProviderCommand: true,
      commandTokens: ["my-acp", "--flag"],
    });
    if (fields.agent_kind === "acp") {
      expect(fields.acp_server).toBe("custom");
      expect(fields.acp_command).toBe("my-acp --flag");
    }
  });

  it("normalizes a blank model to null", () => {
    const fields = buildAgentProfileFields({ ...baseAcp, acpModel: "   " });
    if (fields.agent_kind === "acp") {
      expect(fields.acp_model).toBeNull();
    }
  });
});

describe("buildAgentProfileFields — OpenHands", () => {
  const baseOh = {
    isAcp: false,
    selectedPreset: "custom",
    isDefaultProviderCommand: false,
    commandTokens: [],
    acpModel: "",
    subAgentsEnabled: true,
    toolConcurrencyField: undefined,
    toolConcurrency: "",
  };

  it("passes through enable_sub_agents and omits concurrency when the field is absent", () => {
    expect(buildAgentProfileFields(baseOh)).toEqual({
      agent_kind: "openhands",
      enable_sub_agents: true,
    });
  });

  it("coerces a valid tool_concurrency_limit to a number", () => {
    const fields = buildAgentProfileFields({
      ...baseOh,
      toolConcurrencyField: concurrencyField,
      toolConcurrency: "3",
    });
    if (fields.agent_kind === "openhands") {
      expect(fields.tool_concurrency_limit).toBe(3);
    }
  });

  it("falls back to the schema default (1) when the input is empty, so a clear actually clears (#1571 review)", () => {
    // A blank field coerces to `null`; the field itself is a non-nullable
    // backend int, so an explicit default — not an omitted key — is what
    // actually resets a stored value on an edit-save (the whole-profile merge
    // would otherwise silently keep the old value for an omitted key).
    const fields = buildAgentProfileFields({
      ...baseOh,
      toolConcurrencyField: concurrencyField,
      toolConcurrency: "",
    });
    if (fields.agent_kind === "openhands") {
      expect(fields.tool_concurrency_limit).toBe(1);
    }
  });

  it("falls back to the schema's own default value when the field declares one", () => {
    const fields = buildAgentProfileFields({
      ...baseOh,
      toolConcurrencyField: { ...concurrencyField, default: 2 },
      toolConcurrency: "",
    });
    if (fields.agent_kind === "openhands") {
      expect(fields.tool_concurrency_limit).toBe(2);
    }
  });

  it("throws on a non-numeric concurrency value (schema-driven validation)", () => {
    expect(() =>
      buildAgentProfileFields({
        ...baseOh,
        toolConcurrencyField: concurrencyField,
        toolConcurrency: "abc",
      }),
    ).toThrow();
  });
});
