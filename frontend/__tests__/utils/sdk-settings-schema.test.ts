import { describe, expect, it } from "vitest";

import {
  buildInitialSettingsFormValues,
  buildSdkSettingsPayload,
  buildSdkSettingsPayloadForView,
  getVisibleSettingsSections,
  hasAdvancedSettingsOverrides,
  inferInitialView,
  SPECIALLY_RENDERED_KEYS,
} from "#/utils/sdk-settings-schema";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { Settings } from "#/types/settings";

const BASE_SETTINGS: Settings = {
  ...DEFAULT_SETTINGS,
  agent_settings_schema: {
    model_name: "AgentSettings",
    sections: [
      {
        key: "llm",
        label: "LLM",
        fields: [
          {
            key: "llm.model",
            label: "Model",
            section: "llm",
            section_label: "LLM",
            value_type: "string",
            default: "claude-sonnet-4-20250514",
            choices: [],
            depends_on: [],
            prominence: "critical",
            secret: false,
            required: true,
          },
          {
            key: "llm.api_key",
            label: "API Key",
            section: "llm",
            section_label: "LLM",
            value_type: "string",
            default: null,
            choices: [],
            depends_on: [],
            prominence: "critical",
            secret: true,
            required: false,
          },
          {
            key: "llm.base_url",
            label: "Base URL",
            section: "llm",
            section_label: "LLM",
            value_type: "string",
            default: null,
            choices: [],
            depends_on: [],
            prominence: "critical",
            secret: false,
            required: false,
          },
          {
            key: "llm.litellm_extra_body",
            label: "LiteLLM Extra Body",
            section: "llm",
            section_label: "LLM",
            value_type: "object",
            default: {},
            choices: [],
            depends_on: [],
            prominence: "minor",
            secret: false,
            required: false,
          },
        ],
      },
      {
        key: "verification",
        label: "Verification",
        fields: [
          {
            key: "verification.critic_enabled",
            label: "Enable critic",
            section: "verification",
            section_label: "Verification",
            value_type: "boolean",
            default: true,
            choices: [],
            depends_on: [],
            prominence: "critical",
            secret: false,
            required: true,
          },
          {
            key: "verification.critic_mode",
            label: "Mode",
            section: "verification",
            section_label: "Verification",
            value_type: "string",
            default: "finish_and_message",
            choices: [
              { label: "finish_and_message", value: "finish_and_message" },
              { label: "all_actions", value: "all_actions" },
            ],
            depends_on: ["verification.critic_enabled"],
            prominence: "minor",
            secret: false,
            required: true,
          },
        ],
      },
      {
        key: "general",
        label: "General",
        fields: [
          {
            key: "mcp_config",
            label: "MCP configuration",
            section: "general",
            section_label: "General",
            value_type: "object",
            default: null,
            choices: [],
            depends_on: [],
            prominence: "minor",
            secret: false,
            required: false,
          },
        ],
      },
    ],
  },
  agent_settings: {
    agent: "CodeActAgent",
    llm: {
      api_key: null,
      model: "openai/gpt-4o",
    },
    verification: {
      critic_enabled: false,
      critic_mode: "finish_and_message",
      confirmation_mode: false,
    },
    condenser: {
      enabled: true,
      max_size: 240,
    },
  },
};

describe("sdk settings schema helpers", () => {
  it("builds initial form values from the current settings", () => {
    expect(buildInitialSettingsFormValues(BASE_SETTINGS)).toEqual({
      "llm.api_key": "",
      "llm.base_url": "",
      "llm.litellm_extra_body": "{}",
      "llm.model": "openai/gpt-4o",
      mcp_config: "",
      "verification.critic_enabled": false,
      "verification.critic_mode": "finish_and_message",
    });
  });

  it("detects advanced overrides from non-default values", () => {
    expect(hasAdvancedSettingsOverrides(BASE_SETTINGS)).toBe(false);
    expect(inferInitialView(BASE_SETTINGS)).toBe("basic");

    const withMinorOverride: Settings = {
      ...BASE_SETTINGS,
      agent_settings: {
        ...BASE_SETTINGS.agent_settings,
        verification: {
          ...(BASE_SETTINGS.agent_settings as Record<string, unknown>)
            .verification as Record<string, unknown>,
          critic_mode: "all_actions",
        },
      },
    };
    expect(hasAdvancedSettingsOverrides(withMinorOverride)).toBe(true);
    expect(inferInitialView(withMinorOverride)).toBe("all");
  });

  it("treats empty object value as equivalent to null default (mcp_config serializer artifact)", () => {
    // The backend serialises mcp_config=None as {} via a custom Pydantic
    // serializer, but the schema default is null.  The view should stay
    // "basic" because an empty object is semantically the same as null.
    const withEmptyMcpConfig: Settings = {
      ...BASE_SETTINGS,
      agent_settings: {
        ...BASE_SETTINGS.agent_settings,
        mcp_config: {},
      },
    };
    expect(inferInitialView(withEmptyMcpConfig)).toBe("basic");
  });

  it("filters fields by view tier and excludes specially-rendered keys", () => {
    const values = buildInitialSettingsFormValues(BASE_SETTINGS);

    const basicSections = getVisibleSettingsSections(
      BASE_SETTINGS.agent_settings_schema!,
      values,
      "basic",
    );
    const allBasicFields = basicSections.flatMap((s) => s.fields);
    for (const field of allBasicFields) {
      expect(SPECIALLY_RENDERED_KEYS.has(field.key)).toBe(false);
      expect(field.prominence).toBe("critical");
    }

    const allSections = getVisibleSettingsSections(
      BASE_SETTINGS.agent_settings_schema!,
      { ...values, "verification.critic_enabled": true },
      "all",
    );
    const verificationSection = allSections.find(
      (s) => s.key === "verification",
    );
    expect(verificationSection?.fields).toHaveLength(2);
  });

  it("passes through all fields when excludeKeys is empty", () => {
    const values = buildInitialSettingsFormValues(BASE_SETTINGS);
    const sections = getVisibleSettingsSections(
      BASE_SETTINGS.agent_settings_schema!,
      values,
      "basic",
      new Set(),
    );
    const allFieldKeys = sections.flatMap((s) => s.fields.map((f) => f.key));
    expect(allFieldKeys).toContain("llm.model");
    expect(allFieldKeys).toContain("llm.api_key");
  });

  it("builds a typed payload from dirty schema values", () => {
    const payload = buildSdkSettingsPayload(
      BASE_SETTINGS.agent_settings_schema!,
      {
        ...buildInitialSettingsFormValues(BASE_SETTINGS),
        "verification.critic_enabled": true,
        "llm.api_key": "new-key",
        "llm.litellm_extra_body": JSON.stringify(
          { metadata: { tier: "enterprise" } },
          null,
          2,
        ),
      },
      {
        "verification.critic_enabled": true,
        "llm.api_key": true,
        "llm.litellm_extra_body": true,
        "llm.model": false,
      },
    );

    expect(payload).toEqual({
      llm: {
        api_key: "new-key",
        litellm_extra_body: { metadata: { tier: "enterprise" } },
      },
      verification: { critic_enabled: true },
    });
  });

  it("resets fields outside the selected view back to schema defaults", () => {
    const schema = structuredClone(BASE_SETTINGS.agent_settings_schema!);
    schema.sections[0].fields.push({
      key: "llm.timeout",
      label: "Timeout",
      section: "llm",
      section_label: "LLM",
      value_type: "integer",
      default: 30,
      choices: [],
      depends_on: [],
      prominence: "major",
      secret: false,
      required: false,
    });

    const values = {
      ...buildInitialSettingsFormValues({
        ...BASE_SETTINGS,
        agent_settings_schema: schema,
      }),
      "llm.model": "anthropic/claude-sonnet-4-20250514",
      "llm.timeout": "90",
      "verification.critic_enabled": true,
      "verification.critic_mode": "all_actions",
      "llm.litellm_extra_body": JSON.stringify(
        { metadata: { tier: "enterprise" } },
        null,
        2,
      ),
    };

    const dirty = {
      "llm.model": true,
      "llm.timeout": true,
      "verification.critic_enabled": true,
      "verification.critic_mode": true,
      "llm.litellm_extra_body": true,
    };

    expect(buildSdkSettingsPayloadForView(schema, values, dirty, "basic")).toEqual({
      llm: {
        model: "anthropic/claude-sonnet-4-20250514",
        timeout: 30,
        litellm_extra_body: {},
      },
      mcp_config: null,
      verification: { critic_enabled: true, critic_mode: "finish_and_message" },
    });

    expect(
      buildSdkSettingsPayloadForView(schema, values, dirty, "advanced"),
    ).toEqual({
      llm: {
        model: "anthropic/claude-sonnet-4-20250514",
        timeout: 90,
        litellm_extra_body: {},
      },
      mcp_config: null,
      verification: { critic_enabled: true, critic_mode: "finish_and_message" },
    });

    expect(buildSdkSettingsPayloadForView(schema, values, dirty, "all")).toEqual({
      llm: {
        model: "anthropic/claude-sonnet-4-20250514",
        timeout: 90,
        litellm_extra_body: { metadata: { tier: "enterprise" } },
      },
      verification: { critic_enabled: true, critic_mode: "all_actions" },
    });
  });
});
