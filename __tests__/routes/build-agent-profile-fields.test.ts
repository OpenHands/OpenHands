import { describe, expect, it } from "vitest";
import { buildAgentProfileFields } from "#/routes/agent-settings";
import type { SettingsFieldSchema } from "#/types/settings";

const baseAcp = {
  isAcp: true,
  selectedPreset: "claude-code",
  isDefaultProviderCommand: true,
  commandTokens: ["npx", "-y", "@zed-industries/claude-code-acp"],
  acpModel: "claude-opus-4-8",
  toolConcurrencyField: undefined,
  toolConcurrency: "",
  mcpMode: "standard" as const,
  selectedMcpServers: [] as string[],
  secretsMode: "standard" as const,
  selectedSecrets: [] as string[],
  secretRefsSupportedOnProfile: true,
  toolsMode: "standard" as const,
  selectedTools: [] as string[],
  toolParams: {},
  toolCatalogSupported: true,
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
      mcp_server_refs: null,
      acp_server: "claude-code",
      acp_model: "claude-opus-4-8",
      acp_command: null,
      acp_args: null,
      secret_refs: null,
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
    toolConcurrencyField: undefined,
    toolConcurrency: "",
    mcpMode: "standard" as const,
    selectedMcpServers: [] as string[],
    secretsMode: "standard" as const,
    selectedSecrets: [] as string[],
    secretRefsSupportedOnProfile: true,
  };

  it("no longer emits the retired tool switches", () => {
    // Delegation and LLM switching are picked in `tools` now, so a profile
    // save must not carry a second control over the same tools.
    const fields = buildAgentProfileFields({ ...baseAcp, isAcp: false });

    expect(fields).not.toHaveProperty("enable_sub_agents");
    expect(fields).not.toHaveProperty("enable_switch_llm_tool");
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

describe("buildAgentProfileFields — mcp_server_refs", () => {
  const baseOh = {
    isAcp: false,
    selectedPreset: "custom",
    isDefaultProviderCommand: false,
    commandTokens: [],
    acpModel: "",
    toolsMode: "standard" as const,
    selectedTools: [] as string[],
    toolParams: {},
    toolCatalogSupported: true,
    toolConcurrencyField: undefined,
    toolConcurrency: "",
    mcpMode: "standard" as const,
    selectedMcpServers: [],
    secretsMode: "standard" as const,
    selectedSecrets: [] as string[],
    secretRefsSupportedOnProfile: true,
  };

  it("emits null in standard mode, so the profile inherits every server", () => {
    expect(buildAgentProfileFields(baseOh).mcp_server_refs).toBeNull();
    expect(
      buildAgentProfileFields({ ...baseOh, isAcp: true }).mcp_server_refs,
    ).toBeNull();
  });

  it("emits the selection in custom mode", () => {
    const selected = ["fetch", "playwright"];
    expect(
      buildAgentProfileFields({
        ...baseOh,
        mcpMode: "custom",
        selectedMcpServers: selected,
      }).mcp_server_refs,
    ).toEqual(selected);
  });

  it("distinguishes an empty selection from standard — [] means no servers", () => {
    expect(
      buildAgentProfileFields({ ...baseOh, mcpMode: "custom" }).mcp_server_refs,
    ).toEqual([]);
  });

  it("rides both variants, since the field lives on the profile base", () => {
    const acp = buildAgentProfileFields({
      ...baseOh,
      isAcp: true,
      mcpMode: "custom",
      selectedMcpServers: ["fetch"],
    });
    expect(acp.agent_kind).toBe("acp");
    expect(acp.mcp_server_refs).toEqual(["fetch"]);
  });
});

describe("buildAgentProfileFields — secret scope", () => {
  const base = {
    isAcp: false,
    selectedPreset: "custom",
    isDefaultProviderCommand: false,
    commandTokens: [] as string[],
    acpModel: "",
    toolConcurrencyField: undefined,
    toolConcurrency: "",
    secretsMode: "standard" as const,
    selectedSecrets: [] as string[],
    secretRefsSupportedOnProfile: true,
    mcpMode: "standard" as const,
    selectedMcpServers: [] as string[],
    toolsMode: "standard" as const,
    selectedTools: [] as string[],
    toolParams: {},
    toolCatalogSupported: true,
  };

  it("persists null when every secret is allowed", () => {
    expect(buildAgentProfileFields(base)).toMatchObject({ secret_refs: null });
  });

  it("persists the selection when secrets are scoped", () => {
    const fields = buildAgentProfileFields({
      ...base,
      secretsMode: "custom",
      selectedSecrets: ["DATADOG_API_KEY"],
    });
    expect(fields).toMatchObject({ secret_refs: ["DATADOG_API_KEY"] });
  });

  it("persists an empty list when no secret is selected", () => {
    expect(
      buildAgentProfileFields({ ...base, secretsMode: "custom" }),
    ).toMatchObject({ secret_refs: [] });
  });

  it("rides the ACP variant too — it is a base-model field", () => {
    const fields = buildAgentProfileFields({
      ...base,
      isAcp: true,
      selectedPreset: "claude-code",
      commandTokens: ["npx", "claude-code-acp"],
      secretsMode: "custom",
      selectedSecrets: ["DATADOG_API_KEY"],
    });
    expect(fields).toMatchObject({
      agent_kind: "acp",
      secret_refs: ["DATADOG_API_KEY"],
    });
  });

  it("omits the key on a backend whose profile model predates it", () => {
    const fields = buildAgentProfileFields({
      ...base,
      secretsMode: "custom",
      selectedSecrets: ["DATADOG_API_KEY"],
      secretRefsSupportedOnProfile: false,
    });
    expect(fields).not.toHaveProperty("secret_refs");
  });
});

describe("tool selection", () => {
  const baseOpenHands = { ...baseAcp, isAcp: false };

  it("saves null while the profile follows the server's standard set", () => {
    const fields = buildAgentProfileFields(baseOpenHands);

    expect(fields).toMatchObject({ agent_kind: "openhands", tools: null });
  });

  it("saves the selection with its stored params", () => {
    const fields = buildAgentProfileFields({
      ...baseOpenHands,
      toolsMode: "custom",
      selectedTools: ["terminal", "glob"],
      toolParams: { terminal: { username: "dev" } },
    });

    expect(fields).toMatchObject({
      tools: [
        { name: "terminal", params: { username: "dev" } },
        { name: "glob", params: {} },
      ],
    });
  });

  it("omits tools entirely when the backend serves no catalog", () => {
    // The save overwrites the whole profile, so emitting null here would clear
    // a selection this build never showed the user.
    const fields = buildAgentProfileFields({
      ...baseOpenHands,
      toolCatalogSupported: false,
      toolsMode: "custom",
      selectedTools: ["glob"],
    });

    expect(fields).not.toHaveProperty("tools");
  });

  it("never sends tools for an ACP profile, which owns its own tooling", () => {
    const fields = buildAgentProfileFields({
      ...baseAcp,
      toolsMode: "custom",
      selectedTools: ["glob"],
    });

    expect(fields).not.toHaveProperty("tools");
  });
});
