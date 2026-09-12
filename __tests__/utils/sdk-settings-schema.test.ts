import { describe, expect, it } from "vitest";

import {
  buildInitialSettingsFormValues,
  buildSdkSettingsPayload,
  buildSdkSettingsPayloadForView,
  coerceFieldValue,
  getAgentSettingValue,
  getConversationSettingValue,
  getSettingValue,
  getVisibleSettingsSections,
  hasAdvancedSettings,
  hasAdvancedSettingsOverrides,
  hasMinorSettings,
  inferInitialView,
  isSettingsFieldVisible,
  isValidSettingsSchema,
  normalizeFieldValue,
  SPECIALLY_RENDERED_KEYS,
} from "#/utils/sdk-settings-schema";
import { DEFAULT_SETTINGS } from "#/services/settings";
import {
  Settings,
  SettingsFieldSchema,
  SettingsSchema,
} from "#/types/settings";

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

const getMockField = (
  overrides: Partial<SettingsFieldSchema> = {},
): SettingsFieldSchema => ({
  key: "general.name",
  label: "Name",
  section: "general",
  section_label: "General",
  value_type: "string",
  default: null,
  choices: [],
  depends_on: [],
  prominence: "critical",
  secret: false,
  required: false,
  ...overrides,
});

const getMockSchema = (fields: SettingsFieldSchema[]): SettingsSchema => ({
  model_name: "AgentSettings",
  sections: [{ key: "general", label: "General", fields }],
});

const getSettingsForFields = (
  fields: SettingsFieldSchema[],
  agentSettings: Record<string, never> | Record<string, unknown> | null = {},
): Settings => ({
  ...BASE_SETTINGS,
  agent_settings_schema: getMockSchema(fields),
  agent_settings: agentSettings as Settings["agent_settings"],
});

describe("sdk settings schema helpers", () => {
  it("builds initial form values from the current settings", () => {
    expect(buildInitialSettingsFormValues(BASE_SETTINGS)).toEqual({
      "verification.critic_mode": "finish_and_message",
      "verification.critic_enabled": false,
      "llm.api_key": "",
      "llm.base_url": "",
      "llm.litellm_extra_body": "{}",
      "llm.model": "openai/gpt-4o",
      mcp_config: "",
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
          ...((BASE_SETTINGS.agent_settings as Record<string, unknown>)
            .verification as Record<string, unknown>),
          critic_mode: "all_actions",
        },
      },
    };
    expect(hasAdvancedSettingsOverrides(withMinorOverride)).toBe(true);
    expect(inferInitialView(withMinorOverride)).toBe("all");
  });

  it("treats empty object value as equivalent to null default (mcp_config serializer artifact)", () => {
    // The backend serializes absent mcp_config as {} via a custom Pydantic
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

  it("declares the exact fields owned by purpose-built settings controls", () => {
    expect([...SPECIALLY_RENDERED_KEYS]).toEqual([
      "llm.model",
      "llm.api_key",
      "llm.base_url",
      "llm.auth_type",
      "llm.subscription_vendor",
    ]);
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
          { metadata: { tier: "sample" } },
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
        litellm_extra_body: { metadata: { tier: "sample" } },
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
        { metadata: { tier: "sample" } },
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

    expect(
      buildSdkSettingsPayloadForView(schema, values, dirty, "basic"),
    ).toEqual({
      llm: {
        model: "anthropic/claude-sonnet-4-20250514",
        timeout: 30,
        litellm_extra_body: {},
      },
      verification: { critic_enabled: true, critic_mode: "finish_and_message" },
      mcp_config: null,
    });

    expect(
      buildSdkSettingsPayloadForView(schema, values, dirty, "advanced"),
    ).toEqual({
      llm: {
        model: "anthropic/claude-sonnet-4-20250514",
        timeout: 90,
        litellm_extra_body: {},
      },
      verification: { critic_enabled: true, critic_mode: "finish_and_message" },
      mcp_config: null,
    });

    expect(
      buildSdkSettingsPayloadForView(schema, values, dirty, "all"),
    ).toEqual({
      llm: {
        model: "anthropic/claude-sonnet-4-20250514",
        timeout: 90,
        litellm_extra_body: { metadata: { tier: "sample" } },
      },
      verification: { critic_enabled: true, critic_mode: "all_actions" },
    });
  });

  describe("isValidSettingsSchema", () => {
    it("accepts a schema with an array sections field", () => {
      expect(
        isValidSettingsSchema({
          model_name: "AgentSettings",
          sections: [],
        }),
      ).toBe(true);
    });

    it.each([
      ["null", null],
      ["undefined", undefined],
      ["object without sections", { model_name: "AgentSettings" }],
      [
        "object with non-array sections",
        { model_name: "AgentSettings", sections: "oops" },
      ],
    ])("rejects %s", (_label, value) => {
      expect(isValidSettingsSchema(value as unknown as SettingsSchema)).toBe(
        false,
      );
    });

    it("makes getVisibleSettingsSections tolerate malformed schemas", () => {
      // Regression test for the Vercel preview crash where the schema
      // endpoint resolved with a truthy object that had no `sections`
      // array, causing `.filter` to throw on undefined.
      const malformed = {
        model_name: "AgentSettings",
      } as unknown as SettingsSchema;

      expect(getVisibleSettingsSections(malformed, {}, "basic")).toEqual([]);
    });
  });

  describe("setting value lookup", () => {
    it("reads agent and conversation values without losing falsy values", () => {
      const settings: Settings = {
        ...BASE_SETTINGS,
        agent_settings: {
          llm: { model: "agent-model", enabled: false, retries: 0 },
          empty: "",
        },
        conversation_settings: {
          llm: { model: "conversation-model" },
        },
      };

      expect(getSettingValue(settings, "llm.model")).toBe("agent-model");
      expect(getAgentSettingValue(settings, "llm.enabled")).toBe(false);
      expect(getAgentSettingValue(settings, "llm.retries")).toBe(0);
      expect(getAgentSettingValue(settings, "empty")).toBe("");
      expect(getConversationSettingValue(settings, "llm.model")).toBe(
        "conversation-model",
      );
      expect(
        getSettingValue(settings, "llm.model", "conversation_settings"),
      ).toBe("conversation-model");
    });

    it("returns null for absent roots and paths blocked by scalar values", () => {
      const settings: Settings = {
        ...BASE_SETTINGS,
        agent_settings: {
          llm: "not-an-object",
          nullable: null,
          text: "canvas",
        },
        conversation_settings: null,
      };

      expect(getAgentSettingValue(settings, "missing")).toBeNull();
      expect(getAgentSettingValue(settings, "llm.model")).toBeNull();
      expect(getAgentSettingValue(settings, "nullable.child")).toBeNull();
      expect(getAgentSettingValue(settings, "text.length")).toBeNull();
      expect(getConversationSettingValue(settings, "llm.model")).toBeNull();
    });
  });

  describe("field value normalization", () => {
    it("normalizes choices and missing values for form controls", () => {
      const choiceField = getMockField({
        choices: [{ label: "Zero", value: 0 }],
        default: undefined,
      });

      expect(normalizeFieldValue(choiceField, 0)).toBe("0");
      expect(normalizeFieldValue(choiceField, undefined)).toBe("");
      expect(
        normalizeFieldValue(
          getMockField({
            value_type: "boolean",
            choices: [{ label: "Enabled", value: true }],
          }),
          true,
        ),
      ).toBe("true");
      expect(
        normalizeFieldValue(
          getMockField({ choices: [{ label: "A", value: "a" }] }),
          undefined,
        ),
      ).toBe("");
      expect(
        normalizeFieldValue(getMockField({ default: undefined }), undefined),
      ).toBe("");
      expect(normalizeFieldValue(getMockField(), undefined)).toBe("");
    });

    it("normalizes booleans, structured values, and scalar values", () => {
      expect(
        normalizeFieldValue(
          getMockField({ value_type: "boolean", default: true }),
          null,
        ),
      ).toBe(true);
      expect(
        normalizeFieldValue(
          getMockField({ value_type: "boolean", default: null }),
          undefined,
        ),
      ).toBe(false);
      expect(
        normalizeFieldValue(getMockField({ value_type: "array" }), [1, 2]),
      ).toBe("[\n  1,\n  2\n]");
      expect(
        normalizeFieldValue(getMockField({ value_type: "object" }), {
          nested: true,
        }),
      ).toBe('{\n  "nested": true\n}');
      expect(
        normalizeFieldValue(getMockField({ value_type: "number" }), 12.5),
      ).toBe("12.5");
    });
  });

  describe("initial values and view selection", () => {
    it("returns no form values when the selected schema is absent or malformed", () => {
      expect(
        buildInitialSettingsFormValues({
          ...BASE_SETTINGS,
          agent_settings_schema: null,
        }),
      ).toEqual({});

      const malformed = {
        model_name: "AgentSettings",
      } as unknown as SettingsSchema;
      expect(buildInitialSettingsFormValues(BASE_SETTINGS, malformed)).toEqual(
        {},
      );
    });

    it("builds values from conversation settings and explicit schema overrides", () => {
      const conversationField = getMockField({
        key: "session.enabled",
        value_type: "boolean",
        default: false,
      });
      const settings: Settings = {
        ...BASE_SETTINGS,
        conversation_settings_schema: getMockSchema([conversationField]),
        conversation_settings: { session: { enabled: true } },
      };

      expect(
        buildInitialSettingsFormValues(
          settings,
          undefined,
          "conversation_settings",
        ),
      ).toEqual({ "session.enabled": true });

      const overrideField = getMockField({
        key: "llm.model",
        default: "fallback-model",
      });
      expect(
        buildInitialSettingsFormValues(
          settings,
          getMockSchema([overrideField]),
        ),
      ).toEqual({ "llm.model": "openai/gpt-4o" });
    });

    it("defaults to the basic view when no schema is available", () => {
      expect(
        inferInitialView({
          ...BASE_SETTINGS,
          agent_settings_schema: null,
        }),
      ).toBe("basic");
    });

    it("selects advanced for an overridden major setting", () => {
      const field = getMockField({
        key: "llm.timeout",
        value_type: "integer",
        default: 30,
        prominence: "major",
      });

      expect(
        inferInitialView(
          getSettingsForFields([field], { llm: { timeout: "31" } }),
        ),
      ).toBe("advanced");
    });

    it("selects the most detailed view when minor and major values are overridden", () => {
      const fields = [
        getMockField({
          key: "llm.timeout",
          value_type: "integer",
          default: 30,
          prominence: "major",
        }),
        getMockField({
          key: "llm.extra",
          value_type: "object",
          default: null,
          prominence: "minor",
        }),
      ];

      expect(
        inferInitialView(
          getSettingsForFields(fields, {
            llm: { timeout: 31, extra: { enabled: true } },
          }),
        ),
      ).toBe("all");
    });

    it("selects the view from conversation settings", () => {
      const field = getMockField({
        key: "runtime.mode",
        default: "safe",
        prominence: "major",
      });
      const settings: Settings = {
        ...BASE_SETTINGS,
        conversation_settings_schema: getMockSchema([field]),
        conversation_settings: { runtime: { mode: "fast" } },
      };

      expect(
        inferInitialView(settings, undefined, "conversation_settings"),
      ).toBe("advanced");
    });

    it("treats equivalent boolean and numeric representations as defaults", () => {
      const cases: Array<{
        field: SettingsFieldSchema;
        value: unknown;
      }> = [
        {
          field: getMockField({
            key: "value",
            value_type: "boolean",
            default: true,
            prominence: "major",
          }),
          value: "true",
        },
        {
          field: getMockField({
            key: "value",
            value_type: "boolean",
            default: false,
            prominence: "major",
          }),
          value: "false",
        },
        {
          field: getMockField({
            key: "value",
            value_type: "boolean",
            default: false,
            prominence: "major",
          }),
          value: "",
        },
        {
          field: getMockField({
            key: "value",
            value_type: "boolean",
            default: null,
            prominence: "major",
          }),
          value: null,
        },
        {
          field: getMockField({
            key: "value",
            value_type: "number",
            default: 2,
            prominence: "major",
          }),
          value: "2",
        },
        {
          field: getMockField({
            key: "value",
            value_type: "number",
            default: 2,
            prominence: "major",
          }),
          value: 2,
        },
        {
          field: getMockField({
            key: "value",
            value_type: "number",
            default: null,
            prominence: "major",
          }),
          value: "",
        },
        {
          field: getMockField({
            key: "value",
            value_type: "number",
            default: null,
            prominence: "major",
          }),
          value: "not-a-number",
        },
      ];

      for (const { field, value } of cases) {
        expect(inferInitialView(getSettingsForFields([field], { value }))).toBe(
          "basic",
        );
      }

      const truthyBoolean = getMockField({
        key: "value",
        value_type: "boolean",
        default: false,
        prominence: "major",
      });
      expect(
        inferInitialView(getSettingsForFields([truthyBoolean], { value: 1 })),
      ).toBe("advanced");
    });

    it("preserves boolean, numeric, and scalar comparison boundaries", () => {
      const cases: Array<{
        field: SettingsFieldSchema;
        value: unknown;
        expected: "basic" | "advanced";
      }> = [
        {
          field: getMockField({
            key: "value",
            value_type: "boolean",
            default: true,
            prominence: "major",
          }),
          value: "enabled",
          expected: "basic",
        },
        {
          field: getMockField({
            key: "value",
            value_type: "integer",
            default: 2,
            prominence: "major",
          }),
          value: "02",
          expected: "basic",
        },
        {
          field: getMockField({
            key: "value",
            value_type: "number",
            default: 2,
            prominence: "major",
          }),
          value: "02",
          expected: "basic",
        },
        {
          field: getMockField({
            key: "value",
            value_type: "number",
            default: 0,
            prominence: "major",
          }),
          value: "",
          expected: "advanced",
        },
        {
          field: getMockField({
            key: "value",
            default: "value",
            prominence: "major",
          }),
          value: " value ",
          expected: "advanced",
        },
      ];

      for (const { field, value, expected } of cases) {
        expect(inferInitialView(getSettingsForFields([field], { value }))).toBe(
          expected,
        );
      }
    });

    it("compares structured settings by their parsed content", () => {
      const cases: Array<{
        field: SettingsFieldSchema;
        value: unknown;
        expected: "basic" | "advanced";
      }> = [
        {
          field: getMockField({
            key: "value",
            value_type: "object",
            default: null,
            prominence: "major",
          }),
          value: "{}",
          expected: "basic",
        },
        {
          field: getMockField({
            key: "value",
            value_type: "object",
            default: null,
            prominence: "major",
          }),
          value: "   ",
          expected: "basic",
        },
        {
          field: getMockField({
            key: "value",
            value_type: "object",
            default: { enabled: true },
            prominence: "major",
          }),
          value: '{"enabled":true}',
          expected: "basic",
        },
        {
          field: getMockField({
            key: "value",
            value_type: "object",
            default: null,
            prominence: "major",
          }),
          value: "{invalid",
          expected: "advanced",
        },
        {
          field: getMockField({
            key: "value",
            value_type: "object",
            default: null,
            prominence: "major",
          }),
          value: [],
          expected: "advanced",
        },
        {
          field: getMockField({
            key: "value",
            value_type: "array",
            default: [1],
            prominence: "major",
          }),
          value: " [1] ",
          expected: "basic",
        },
        {
          field: getMockField({
            key: "value",
            value_type: "array",
            default: null,
            prominence: "major",
          }),
          value: [],
          expected: "advanced",
        },
      ];

      for (const { field, value, expected } of cases) {
        expect(inferInitialView(getSettingsForFields([field], { value }))).toBe(
          expected,
        );
      }
    });

    it("does not collapse non-object structured values into an empty object", () => {
      const cases: Array<{
        field: SettingsFieldSchema;
        value: unknown;
      }> = [
        {
          field: getMockField({
            key: "value",
            value_type: "array",
            default: null,
            prominence: "major",
          }),
          value: {},
        },
        {
          field: getMockField({
            key: "value",
            value_type: "object",
            default: null,
            prominence: "major",
          }),
          value: 1,
        },
        {
          field: getMockField({
            key: "value",
            value_type: "array",
            default: null,
            prominence: "major",
          }),
          value: "{}",
        },
        {
          field: getMockField({
            key: "value",
            value_type: "object",
            default: null,
            prominence: "major",
          }),
          value: "1",
        },
      ];

      for (const { field, value } of cases) {
        expect(inferInitialView(getSettingsForFields([field], { value }))).toBe(
          "advanced",
        );
      }
    });

    it("keeps stringified null distinct from an absent object setting", () => {
      const field = getMockField({
        key: "value",
        value_type: "object",
        default: null,
        prominence: "major",
      });

      expect(
        inferInitialView(getSettingsForFields([field], { value: "null" })),
      ).toBe("advanced");
    });

    it("compares invalid structured text after trimming whitespace", () => {
      const field = getMockField({
        key: "value",
        value_type: "object",
        default: "{invalid",
        prominence: "major",
      });

      expect(
        inferInitialView(
          getSettingsForFields([field], { value: "  {invalid  " }),
        ),
      ).toBe("basic");
    });

    it("compares nullable and non-string scalar settings", () => {
      const nullable = getMockField({
        key: "value",
        default: null,
        prominence: "major",
      });
      expect(inferInitialView(getSettingsForFields([nullable]))).toBe("basic");

      const numericScalar = getMockField({
        key: "value",
        default: "1",
        prominence: "major",
      });
      expect(
        inferInitialView(getSettingsForFields([numericScalar], { value: 2 })),
      ).toBe("advanced");
    });
  });

  describe("field coercion", () => {
    it("coerces boolean controls and rejects ambiguous boolean text", () => {
      const field = getMockField({
        label: "Enabled",
        value_type: "boolean",
      });

      expect(coerceFieldValue(field, true)).toBe(true);
      expect(coerceFieldValue(field, false)).toBe(false);
      expect(coerceFieldValue(field, "   ")).toBeNull();
      expect(coerceFieldValue(field, " TrUe ")).toBe(true);
      expect(coerceFieldValue(field, " FALSE ")).toBe(false);
      expect(() => coerceFieldValue(field, "sometimes")).toThrow(
        "Expected a boolean value, received: sometimes",
      );
    });

    it("coerces numeric controls and validates numeric shape", () => {
      const integerField = getMockField({
        key: "retries",
        label: "Retries",
        value_type: "integer",
      });
      const numberField = getMockField({
        key: "ratio",
        label: "Ratio",
        value_type: "number",
      });

      expect(coerceFieldValue(integerField, "  ")).toBeNull();
      expect(coerceFieldValue(integerField, "3")).toBe(3);
      expect(coerceFieldValue(numberField, "2.5")).toBe(2.5);
      expect(() => coerceFieldValue(numberField, "many")).toThrow(
        "Expected a numeric value, received: many",
      );
      expect(() => coerceFieldValue(integerField, "1.5")).toThrow(
        "Expected an integer value, received: 1.5",
      );
    });

    it("enforces configured minimum and maximum values", () => {
      const concurrencyField = getMockField({
        key: "tool_concurrency_limit",
        label: "Tool concurrency",
        value_type: "integer",
      });
      const temperatureField = getMockField({
        key: "llm.temperature",
        label: "Temperature",
        value_type: "number",
      });

      expect(() => coerceFieldValue(concurrencyField, "0")).toThrow(
        "Tool concurrency must be at least 1",
      );
      expect(coerceFieldValue(concurrencyField, "1")).toBe(1);
      expect(coerceFieldValue(temperatureField, "0")).toBe(0);
      expect(coerceFieldValue(temperatureField, "2")).toBe(2);
      expect(() => coerceFieldValue(temperatureField, "2.1")).toThrow(
        "Temperature must be at most 2",
      );
    });

    it("coerces JSON arrays and reports invalid array input", () => {
      const field = getMockField({
        label: "Tags",
        value_type: "array",
      });

      expect(coerceFieldValue(field, " ")).toBeNull();
      expect(coerceFieldValue(field, '["one", 2]')).toEqual(["one", 2]);
      expect(() => coerceFieldValue(field, "not-json")).toThrow(
        "Invalid JSON for Tags",
      );
      expect(() => coerceFieldValue(field, '{"one":1}')).toThrow(
        "Tags must be a JSON array",
      );
    });

    it("coerces JSON objects and rejects every non-object JSON shape", () => {
      const field = getMockField({
        label: "Metadata",
        value_type: "object",
      });

      expect(coerceFieldValue(field, '{"tier":"sample"}')).toEqual({
        tier: "sample",
      });
      expect(() => coerceFieldValue(field, "null")).toThrow(
        "Metadata must be a JSON object",
      );
      expect(() => coerceFieldValue(field, "[]")).toThrow(
        "Metadata must be a JSON object",
      );
      expect(() => coerceFieldValue(field, '"text"')).toThrow(
        "Metadata must be a JSON object",
      );
    });

    it("clears empty public strings while preserving empty secrets", () => {
      expect(coerceFieldValue(getMockField(), "")).toBeNull();
      expect(coerceFieldValue(getMockField({ secret: true }), "")).toBe("");
      expect(coerceFieldValue(getMockField(), "value")).toBe("value");
    });
  });

  describe("payload construction", () => {
    it("returns an empty payload for malformed schemas", () => {
      const malformed = {
        model_name: "AgentSettings",
      } as unknown as SettingsSchema;

      expect(buildSdkSettingsPayload(malformed, {}, {})).toEqual({});
      expect(
        buildSdkSettingsPayloadForView(malformed, {}, {}, "basic"),
      ).toEqual({});
    });

    it("replaces conflicting scalar, null, and array parents with nested values", () => {
      const fields = [
        getMockField({ key: "scalar" }),
        getMockField({ key: "scalar.child" }),
        getMockField({ key: "nullable" }),
        getMockField({ key: "nullable.child" }),
        getMockField({ key: "items", value_type: "array" }),
        getMockField({ key: "items.child" }),
      ];
      const values = {
        scalar: "flat",
        "scalar.child": "nested",
        nullable: "",
        "nullable.child": "nested",
        items: '["first"]',
        "items.child": "nested",
      };
      const dirty = Object.fromEntries(
        fields.map((field) => [field.key, true]),
      );

      expect(
        buildSdkSettingsPayload(getMockSchema(fields), values, dirty),
      ).toEqual({
        scalar: { child: "nested" },
        nullable: { child: "nested" },
        items: { child: "nested" },
      });
    });

    it("keeps clean fields out of the payload", () => {
      const field = getMockField({ key: "name" });
      expect(
        buildSdkSettingsPayload(
          getMockSchema([field]),
          { name: "Ada" },
          { name: false },
        ),
      ).toEqual({});
    });

    it("resets an out-of-view field with no default to null", () => {
      const field = getMockField({
        key: "optional",
        default: undefined,
        prominence: "major",
      });

      expect(
        buildSdkSettingsPayloadForView(
          getMockSchema([field]),
          { optional: "changed" },
          { optional: true },
          "basic",
        ),
      ).toEqual({ optional: null });
    });
  });

  describe("field and section visibility", () => {
    it("requires every declared dependency to be enabled", () => {
      expect(isSettingsFieldVisible(getMockField(), {})).toBe(true);
      const dependent = getMockField({ depends_on: ["one", "two"] });
      expect(isSettingsFieldVisible(dependent, { one: true, two: true })).toBe(
        true,
      );
      expect(isSettingsFieldVisible(dependent, { one: true, two: false })).toBe(
        false,
      );
      expect(isSettingsFieldVisible(dependent, { one: true })).toBe(false);
    });

    it("keeps only fields allowed by exclusions, view, and dependencies", () => {
      const visible = getMockField({ key: "visible" });
      const excluded = getMockField({ key: "excluded" });
      const major = getMockField({ key: "major", prominence: "major" });
      const dependent = getMockField({
        key: "dependent",
        depends_on: ["enabled"],
      });
      const schema: SettingsSchema = {
        model_name: "AgentSettings",
        sections: [
          {
            key: "general",
            label: "General",
            fields: [visible, excluded, major, dependent],
          },
          {
            key: "empty-after-filtering",
            label: "Hidden",
            fields: [getMockField({ key: "also-excluded" })],
          },
        ],
      };

      expect(
        getVisibleSettingsSections(
          schema,
          { enabled: false },
          "basic",
          new Set(["excluded", "also-excluded"]),
        ),
      ).toEqual([
        {
          key: "general",
          label: "General",
          fields: [visible],
        },
      ]);
    });

    it("reports whether advanced and minor tiers exist", () => {
      const schema = getMockSchema([
        getMockField({ key: "critical" }),
        getMockField({ key: "advanced", prominence: "major" }),
        getMockField({ key: "minor", prominence: "minor" }),
      ]);

      expect(hasAdvancedSettings(null)).toBe(false);
      expect(hasMinorSettings(null)).toBe(false);
      expect(hasAdvancedSettings(getMockSchema([]))).toBe(false);
      expect(hasMinorSettings(getMockSchema([]))).toBe(false);
      expect(hasAdvancedSettings(schema)).toBe(true);
      expect(hasMinorSettings(schema)).toBe(true);
    });

    it("distinguishes each optional prominence tier from critical fields", () => {
      const criticalOnly = getMockSchema([
        getMockField({ key: "critical", prominence: "critical" }),
      ]);
      const majorOnly = getMockSchema([
        getMockField({ key: "major", prominence: "major" }),
      ]);
      const minorOnly = getMockSchema([
        getMockField({ key: "minor", prominence: "minor" }),
      ]);

      expect(hasAdvancedSettings(criticalOnly)).toBe(false);
      expect(hasAdvancedSettings(majorOnly)).toBe(true);
      expect(hasMinorSettings(criticalOnly)).toBe(false);
      expect(hasMinorSettings(minorOnly)).toBe(true);
    });
  });
});
