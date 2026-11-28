import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { normalizeFieldValue, coerceFieldValue } from "./sdk-settings-schema";
import { SettingsFieldSchema } from "#/types/settings";
import { SchemaField } from "#/components/features/settings/sdk-settings/schema-field";

// The global test mock returns the key and drops interpolation values. The cost
// hint is only observable through its interpolated value, so keep it here.
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, unknown>) =>
      key === "SCHEMA$LLM$COST_PER_TOKEN_HINT"
        ? `= $${String(options?.value ?? "")} per token`
        : key,
    i18n: { language: "en", exists: () => false },
  }),
}));

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

describe("LLM cost per million tokens formatting", () => {
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

  it("round-trips a stored value with more than six significant digits without losing precision", () => {
    const storedPerTokenValue = 0.0000000387654321;

    const displayValue = normalizeFieldValue(
      inputCostField,
      storedPerTokenValue,
    );
    const roundTripped = coerceFieldValue(
      inputCostField,
      displayValue as string,
    );

    expect(displayValue).toBe("0.0387654321");
    expect(roundTripped).toBe(storedPerTokenValue);
  });

  it("never returns a string containing exponent notation when normalizing", () => {
    const storedValues = [1e-16, 1e-12, 0.0000000387654321, 1e21];

    for (const storedValue of storedValues) {
      const displayValue = normalizeFieldValue(inputCostField, storedValue);
      expect(typeof displayValue).toBe("string");
      expect(displayValue).not.toMatch(/[eE]/);
    }
  });
});

describe("SchemaField LLM cost hint", () => {
  function renderCostField(value: string) {
    render(
      React.createElement(SchemaField, {
        field: inputCostField,
        value,
        isDisabled: false,
        onChange: () => {},
      }),
    );
    return screen.getByText(/per token/);
  }

  it("shows a per-token value of 0.0000025 in plain decimal, never exponent notation", () => {
    // 2.5 $/M corresponds to a stored per-token value of 0.0000025.
    const hint = renderCostField("2.5");

    expect(hint).toHaveTextContent("0.0000025");
    expect(hint).not.toHaveTextContent(/e[+-]/);
  });

  it("never uses exponent notation for very small per-token values", () => {
    // 0.25 $/M corresponds to 2.5e-7 per token, which used to render as "2.5e-7".
    const hint = renderCostField("0.25");

    expect(hint).toHaveTextContent("0.00000025");
    expect(hint).not.toHaveTextContent(/e[+-]/);
  });
});
