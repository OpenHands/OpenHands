import { describe, it, expect } from "vitest";
import type { SettingsSchema, SettingsFieldSchema } from "#/types/settings";
import {
  buildSdkSettingsPayload,
  buildSdkSettingsPayloadForView,
  coerceFieldValue,
} from "./sdk-settings-schema";
import { resolveNumericConstraints } from "./sdk-settings-field-metadata";

/**
 * Minimal schema factory: two sections, each with one field at each of the
 * three prominence tiers, so every view (basic / advanced / all) has exactly
 * one visible field and two hidden ones.
 */
function makeSchema(): SettingsSchema {
  const field = (
    key: string,
    prominence: SettingsFieldSchema["prominence"],
    opts: Partial<SettingsFieldSchema> = {},
  ): SettingsFieldSchema => ({
    key,
    label: key,
    description: null,
    section: key.split(".")[0] ?? key,
    section_label: key.split(".")[0] ?? key,
    value_type: "integer",
    default: 5,
    choices: [],
    depends_on: [],
    prominence,
    secret: false,
    required: false,
    ...opts,
  });

  return {
    model_name: "test",
    sections: [
      {
        key: "llm",
        label: "LLM",
        fields: [
          field("llm.critical_field", "critical", { default: 1 }),
          field("llm.major_field", "major", { default: 10 }),
          field("llm.minor_field", "minor", { default: 100 }),
        ],
      },
    ],
  };
}

describe("buildSdkSettingsPayloadForView", () => {
  it("does NOT reset fields outside the current view to their defaults", () => {
    const schema = makeSchema();
    // User changed all three fields (e.g. while on "all" view).
    const values = {
      "llm.critical_field": "2",
      "llm.major_field": "20",
      "llm.minor_field": "200",
    };
    const dirty = {
      "llm.critical_field": true,
      "llm.major_field": true,
      "llm.minor_field": true,
    };

    // User later switches back to "basic" and saves.
    // BUGGY BEHAVIOUR (pre-fix): the function would force llm.major_field and
    // llm.minor_field back to their schema defaults (10 and 100) in the
    // payload, so the next load would show those values instead of 20/200.
    const payload = buildSdkSettingsPayloadForView(
      schema,
      values,
      dirty,
      "basic",
    );

    // FIXED: every dirty field the user actually touched is present at its
    // current value. Nothing gets silently reset to a default.
    expect(payload).toEqual({
      llm: {
        critical_field: 2,
        major_field: 20,
        minor_field: 200,
      },
    });
  });

  it("omits fields that were never dirty, regardless of view", () => {
    const schema = makeSchema();
    const values = {
      "llm.critical_field": "2",
      "llm.major_field": "10", // same as default, never edited
      "llm.minor_field": "200",
    };
    const dirty = {
      "llm.critical_field": true,
      "llm.minor_field": true,
    };

    const payload = buildSdkSettingsPayloadForView(
      schema,
      values,
      dirty,
      "basic",
    );

    expect(payload).toEqual({
      llm: {
        critical_field: 2,
        minor_field: 200,
      },
    });
    // major_field was never dirty → absent from payload → backend leaves it
    // untouched. No spurious default-injection.
    expect(payload.llm).not.toHaveProperty("major_field");
  });
});

describe("resolveNumericConstraints (SDK-first, metadata-fallback)", () => {
  it("returns SDK-supplied min/max when the schema exposes them", () => {
    const field: SettingsFieldSchema = {
      key: "some.custom.field",
      label: "Custom",
      description: null,
      section: "x",
      section_label: "X",
      value_type: "integer",
      default: 5,
      choices: [],
      depends_on: [],
      prominence: "minor",
      secret: false,
      required: false,
      constraints: { min: 2, max: 20, step: 1 },
    };
    expect(resolveNumericConstraints(field)).toEqual({
      min: 2,
      max: 20,
      step: 1,
      exclusive_min: undefined,
      exclusive_max: undefined,
    });
  });

  it("falls back to the frontend hardcoded metadata map for older servers", () => {
    // "llm.top_p" is hardcoded in FIELD_METADATA as min:0, max:1, step:0.01.
    const field: SettingsFieldSchema = {
      key: "llm.top_p",
      label: "Top P",
      description: null,
      section: "llm",
      section_label: "LLM",
      value_type: "number",
      default: 0.9,
      choices: [],
      depends_on: [],
      prominence: "major",
      secret: false,
      required: false,
      // constraints is absent → use metadata fallback.
    };
    expect(resolveNumericConstraints(field)).toEqual({
      min: 0,
      max: 1,
      step: 0.01,
    });
  });

  it("SDK values override hardcoded metadata when both exist", () => {
    const field: SettingsFieldSchema = {
      key: "llm.top_p",
      label: "Top P",
      description: null,
      section: "llm",
      section_label: "LLM",
      value_type: "number",
      default: 0.9,
      choices: [],
      depends_on: [],
      prominence: "major",
      secret: false,
      required: false,
      // Newer SDK widens the range.
      constraints: { min: -0.5, max: 1.5 },
    };
    const resolved = resolveNumericConstraints(field);
    expect(resolved?.min).toBe(-0.5); // SDK wins over hardcoded 0
    expect(resolved?.max).toBe(1.5); // SDK wins over hardcoded 1
    expect(resolved?.step).toBe(0.01); // metadata fills in step
  });

  it("returns undefined when neither source supplies constraints", () => {
    const field: SettingsFieldSchema = {
      key: "no_constraints.field",
      label: "X",
      description: null,
      section: "x",
      section_label: "X",
      value_type: "integer",
      default: 0,
      choices: [],
      depends_on: [],
      prominence: "minor",
      secret: false,
      required: false,
    };
    expect(resolveNumericConstraints(field)).toBeUndefined();
  });

  it("carries exclusive_min / exclusive_max when SDK supplies them", () => {
    const field: SettingsFieldSchema = {
      key: "strict.field",
      label: "Strict",
      description: null,
      section: "x",
      section_label: "X",
      value_type: "number",
      default: 0,
      choices: [],
      depends_on: [],
      prominence: "minor",
      secret: false,
      required: false,
      constraints: { min: 0, exclusive_min: true, max: 1, exclusive_max: true },
    };
    const resolved = resolveNumericConstraints(field);
    expect(resolved?.min).toBe(0);
    expect(resolved?.exclusive_min).toBe(true);
    expect(resolved?.exclusive_max).toBe(true);
  });
});

describe("coerceFieldValue respects SDK constraints", () => {
  const numericField = (
    key: string,
    constraints: NonNullable<SettingsFieldSchema["constraints"]>,
  ): SettingsFieldSchema => ({
    key,
    label: key,
    description: null,
    section: "x",
    section_label: "X",
    value_type: "number",
    default: 0,
    choices: [],
    depends_on: [],
    prominence: "minor",
    secret: false,
    required: false,
    constraints,
  });

  it("accepts in-range values", () => {
    const field = numericField("x", { min: 0, max: 1 });
    expect(coerceFieldValue(field, "0.5")).toBeCloseTo(0.5);
  });

  it("rejects values below an inclusive min", () => {
    const field = numericField("x", { min: 0, max: 1 });
    expect(() => coerceFieldValue(field, "-0.1")).toThrow(/at least 0/);
  });

  it("rejects values at or below an exclusive min", () => {
    const field = numericField("x", { min: 0, exclusive_min: true, max: 1 });
    expect(() => coerceFieldValue(field, "0")).toThrow(/greater than 0/);
    expect(() => coerceFieldValue(field, "-0.1")).toThrow(/greater than 0/);
    expect(coerceFieldValue(field, "0.0001")).toBeCloseTo(0.0001);
  });

  it("rejects values at or above an exclusive max", () => {
    const field = numericField("x", { min: 0, max: 1, exclusive_max: true });
    expect(() => coerceFieldValue(field, "1")).toThrow(/less than 1/);
    expect(() => coerceFieldValue(field, "1.1")).toThrow(/less than 1/);
    expect(coerceFieldValue(field, "0.9999")).toBeCloseTo(0.9999);
  });
});

describe("round-trip: hidden-field persistence (#18049)", () => {
  /**
   * This is the exact scenario described in the issue: a user edits
   * "minor"-prominence fields while on the "all" view, then later edits a
   * "critical" field on the "basic" view. Before the fix, the basic-view
   * save silently reset every hidden (major/minor) field back to its schema
   * default — the most destructive sort of data loss because the change
   * happens without any user action on those fields.
   */
  it("saving a basic-view edit preserves prior hidden-field edits", () => {
    const schema: SettingsSchema = {
      model_name: "test",
      sections: [
        {
          key: "llm",
          label: "LLM",
          fields: [
            {
              key: "llm.model",
              label: "Model",
              description: null,
              section: "llm",
              section_label: "LLM",
              value_type: "string",
              default: "default-model",
              choices: [],
              depends_on: [],
              prominence: "critical",
              secret: false,
              required: false,
            },
            {
              key: "llm.temperature",
              label: "Temperature",
              description: null,
              section: "llm",
              section_label: "LLM",
              value_type: "number",
              default: 0.7,
              choices: [],
              depends_on: [],
              prominence: "major",
              secret: false,
              required: false,
            },
            {
              key: "llm.top_p",
              label: "Top P",
              description: null,
              section: "llm",
              section_label: "LLM",
              value_type: "number",
              default: 0.9,
              choices: [],
              depends_on: [],
              prominence: "minor",
              secret: false,
              required: false,
            },
          ],
        },
      ],
    };

    // Step 1: user edits everything while on "all" view — every field is dirty.
    const valuesAll = {
      "llm.model": "gpt-4",
      "llm.temperature": "0.3",
      "llm.top_p": "0.5",
    };
    // Step 2: later, the user is on "basic" and changes only the model.
    const valuesBasic = { ...valuesAll };
    const dirtyBasic = { "llm.model": true };

    // Step 3: save from basic view.
    const patch = buildSdkSettingsPayloadForView(
      schema,
      valuesBasic,
      dirtyBasic,
      "basic",
    );

    // The patch must contain *only* the dirty field — a PATCH diff — and must
    // NOT silently reinject llm.temperature=0.7 and llm.top_p=0.9 (defaults),
    // which would overwrite the user's prior edits on the next round-trip.
    expect(patch).toEqual({
      llm: {
        model: "gpt-4",
      },
    });
    // The proof: hidden-field defaults are completely absent.
    expect(patch.llm).not.toHaveProperty("temperature");
    expect(patch.llm).not.toHaveProperty("top_p");

    // Sanity check: the dirty-only helper agrees.
    expect(buildSdkSettingsPayload(schema, valuesBasic, dirtyBasic)).toEqual(
      patch,
    );
  });
});
