import { describe, it, expect } from "vitest";
import { normalizeFieldValue, coerceFieldValue } from "./sdk-settings-schema";
import { SettingsFieldSchema } from "#/types/settings";

describe("LLM cost per million tokens formatting", () => {
  const inputCostField = {
    key: "llm.input_cost_per_token",
    label: "Input Cost Per Token",
    value_type: "number",
    default: null,
    choices: [],
    required: false,
    secret: false,
    prominence: "advanced",
    depends_on: [],
  } as unknown as SettingsFieldSchema;

  it("multiplies per-token values by 1e6 and removes exponent notation when normalizing", () => {
    expect(normalizeFieldValue(inputCostField, 0.000000038)).toBe("0.038");
    expect(normalizeFieldValue(inputCostField, 0.0000025)).toBe("2.5");
  });

  it("divides $/M inputs by 1e6 when coercing for save", () => {
    expect(coerceFieldValue(inputCostField, "0.038")).toBe(0.000000038);
    expect(coerceFieldValue(inputCostField, "2.5")).toBe(0.0000025);
  });

  it("throws a guardrail error if a user pastes a per-token value into the $/M field", () => {
    expect(() => coerceFieldValue(inputCostField, "0.0000025")).toThrow(
      "Looks like a per-token price. Did you mean 2.5?",
    );
  });
});
