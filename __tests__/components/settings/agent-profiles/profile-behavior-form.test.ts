import { describe, expect, it } from "vitest";
import {
  fromBehaviorFormValues,
  toBehaviorFormValues,
} from "#/components/features/settings/agent-profiles/profile-behavior-form";
import type { SettingsFieldSchema, SettingsSchema } from "#/types/settings";

const field = (
  key: string,
  value_type: SettingsFieldSchema["value_type"],
  defaultValue: SettingsFieldSchema["default"],
): SettingsFieldSchema =>
  ({
    key,
    label: key,
    section: key.split(".")[0],
    section_label: key.split(".")[0],
    value_type,
    default: defaultValue,
    choices: [],
    depends_on: [],
    prominence: "critical",
    secret: false,
    required: false,
  }) as SettingsFieldSchema;

const SCHEMA: SettingsSchema = {
  model_name: "AgentSettings",
  sections: [
    {
      key: "verification",
      label: "Verification",
      fields: [
        field("verification.critic_enabled", "boolean", false),
        field("verification.critic_threshold", "number", 0.6),
        field("verification.critic_api_key", "string", null),
      ],
    },
  ],
};

describe("profile behavior form", () => {
  it("round-trips a stored section, filling a missing field with its default", () => {
    const values = toBehaviorFormValues(SCHEMA, "verification", {
      critic_enabled: true,
    });

    expect(fromBehaviorFormValues(SCHEMA, "verification", values)).toEqual({
      critic_enabled: true,
      critic_threshold: 0.6,
    });
  });

  it("never reads or writes the shared critic API key", () => {
    const values = toBehaviorFormValues(SCHEMA, "verification", {
      critic_api_key: "sk-stored",
    });

    expect(values).not.toHaveProperty("verification.critic_api_key");
    expect(
      fromBehaviorFormValues(SCHEMA, "verification", {
        ...values,
        "verification.critic_api_key": "sk-typed",
      }),
    ).not.toHaveProperty("critic_api_key");
  });
});
