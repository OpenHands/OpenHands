import {
  SettingProminence,
  Settings,
  SettingsFieldSchema,
  SettingsSchema,
  SettingsSectionSchema,
  SettingsValue,
} from "#/types/settings";
import { resolveNumericConstraints } from "#/utils/sdk-settings-field-metadata";
import {
  LLM_AUTH_TYPE_KEY,
  LLM_SUBSCRIPTION_VENDOR_KEY,
} from "#/constants/llm-subscription";

export type SettingsFormValues = Record<string, string | boolean>;
export type SettingsDirtyState = Record<string, boolean>;
export type SdkSettingsPayload = Record<string, SettingsValue>;
export type SettingsValueSource = "agent_settings" | "conversation_settings";

export type SettingsView = "basic" | "advanced" | "all";

/** Fields that are rendered by purpose-built components instead of the
 *  generic `SchemaField` renderer. */
export const SPECIALLY_RENDERED_KEYS = new Set([
  "llm.model",
  "llm.api_key",
  "llm.base_url",
  LLM_AUTH_TYPE_KEY,
  LLM_SUBSCRIPTION_VENDOR_KEY,
]);

/** Prominence tiers visible at each view level. */
const VIEW_PROMINENCES: Record<SettingsView, Set<SettingProminence>> = {
  basic: new Set<SettingProminence>(["critical"]),
  advanced: new Set<SettingProminence>(["critical", "major"]),
  all: new Set<SettingProminence>(["critical", "major", "minor"]),
};

/**
 * True when `schema` looks like a usable `SettingsSchema` — i.e. an
 * object with an array `sections` field. Guards every helper in this
 * module against malformed/empty schema responses (e.g. when the
 * frontend ends up pointing at a host that does not actually serve
 * `/api/settings/agent-schema`, such as an unconfigured Vercel preview
 * origin that returns the React Router SPA shell for arbitrary
 * `/api/*` paths). Without this check, `schema.sections.filter(...)`
 * inside `SdkSectionPage` blows up with
 * `Cannot read properties of undefined (reading 'filter')` and React
 * Router escalates to a full-screen error page.
 */
export function isValidSettingsSchema(
  schema: SettingsSchema | null | undefined,
): schema is SettingsSchema {
  return !!schema && Array.isArray((schema as SettingsSchema).sections);
}

function getSchemaFields(schema: SettingsSchema): SettingsFieldSchema[] {
  if (!isValidSettingsSchema(schema)) return [];
  return schema.sections.flatMap((section) => section.fields);
}

/** Traverse a nested object using a dotted key path (e.g. "llm.model"). */
function lookupDotted(
  obj: Record<string, unknown> | null | undefined,
  key: string,
): unknown {
  if (!obj) return undefined;
  const parts = key.split(".");
  let current: unknown = obj;
  for (const part of parts) {
    if (current == null || typeof current !== "object") return undefined;
    current = (current as Record<string, unknown>)[part];
  }
  return current;
}

/** Set a value in a nested object at a dotted key path (e.g. "llm.model"). */
function setDotted(
  obj: Record<string, unknown>,
  key: string,
  value: unknown,
): void {
  const parts = key.split(".");
  let current: Record<string, unknown> = obj;
  for (let i = 0; i < parts.length - 1; i += 1) {
    const part = parts[i];
    if (
      current[part] == null ||
      typeof current[part] !== "object" ||
      Array.isArray(current[part])
    ) {
      current[part] = {};
    }
    current = current[part] as Record<string, unknown>;
  }
  current[parts[parts.length - 1]] = value;
}

export function getSettingValue(
  settings: Settings,
  key: string,
  source: SettingsValueSource = "agent_settings",
): SettingsValue {
  return (lookupDotted(settings[source] as Record<string, unknown>, key) ??
    null) as SettingsValue;
}

export function getAgentSettingValue(
  settings: Settings,
  key: string,
): SettingsValue {
  return getSettingValue(settings, key, "agent_settings");
}

function isChoiceField(field: SettingsFieldSchema): boolean {
  return field.choices.length > 0;
}

function isCriticalField(field: SettingsFieldSchema): boolean {
  return field.prominence === "critical";
}

function isMinorField(field: SettingsFieldSchema): boolean {
  return field.prominence === "minor";
}

export function normalizeFieldValue(
  field: SettingsFieldSchema,
  rawValue: unknown,
): string | boolean {
  const resolvedValue = rawValue ?? field.default;

  if (isChoiceField(field)) {
    return resolvedValue === null || resolvedValue === undefined
      ? ""
      : String(resolvedValue);
  }

  if (field.value_type === "boolean") {
    return Boolean(resolvedValue ?? false);
  }

  if (resolvedValue === null || resolvedValue === undefined) {
    return "";
  }

  if (field.value_type === "array" || field.value_type === "object") {
    return JSON.stringify(resolvedValue, null, 2);
  }

  return String(resolvedValue);
}

function stableJsonStringify(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }

  if (Array.isArray(value)) {
    return `[${value.map(stableJsonStringify).join(",")}]`;
  }

  const sortedEntries = Object.entries(value as Record<string, unknown>).sort(
    ([leftKey], [rightKey]) => leftKey.localeCompare(rightKey),
  );

  return `{${sortedEntries
    .map(
      ([key, entryValue]) =>
        `${JSON.stringify(key)}:${stableJsonStringify(entryValue)}`,
    )
    .join(",")}}`;
}

export function normalizeComparableValue(
  field: SettingsFieldSchema,
  rawValue: unknown,
): boolean | number | string | null {
  if (rawValue === undefined) {
    return null;
  }

  if (field.value_type === "boolean") {
    if (typeof rawValue === "string") {
      if (rawValue === "true") {
        return true;
      }
      if (rawValue === "false") {
        return false;
      }
    }
    if (rawValue === null) {
      return null;
    }
    return Boolean(rawValue);
  }

  if (field.value_type === "integer" || field.value_type === "number") {
    if (rawValue === "" || rawValue === null) {
      return null;
    }

    const parsedValue =
      typeof rawValue === "number" ? rawValue : Number(String(rawValue));
    return Number.isNaN(parsedValue) ? null : parsedValue;
  }

  if (field.value_type === "array" || field.value_type === "object") {
    if (rawValue === null) {
      return null;
    }

    // Treat empty objects as null so that serializer artifacts
    // (e.g. mcp_config: {} vs schema default null) don't trigger
    // a spurious view escalation in inferInitialView.
    if (
      field.value_type === "object" &&
      typeof rawValue === "object" &&
      !Array.isArray(rawValue) &&
      Object.keys(rawValue as Record<string, unknown>).length === 0
    ) {
      return null;
    }

    if (typeof rawValue === "string") {
      const trimmedValue = rawValue.trim();
      if (!trimmedValue) {
        return null;
      }
      try {
        const parsed: unknown = JSON.parse(trimmedValue);
        // Also normalise stringified empty objects
        if (
          field.value_type === "object" &&
          parsed !== null &&
          typeof parsed === "object" &&
          !Array.isArray(parsed) &&
          Object.keys(parsed as Record<string, unknown>).length === 0
        ) {
          return null;
        }
        return stableJsonStringify(parsed);
      } catch {
        return trimmedValue;
      }
    }

    return stableJsonStringify(rawValue);
  }

  if (rawValue === null) {
    return null;
  }

  return String(rawValue);
}

export function buildInitialSettingsFormValues(
  settings: Settings,
  schemaOverride?: SettingsSchema | null,
  source: SettingsValueSource = "agent_settings",
): SettingsFormValues {
  const schema =
    schemaOverride ??
    (source === "conversation_settings"
      ? settings.conversation_settings_schema
      : settings.agent_settings_schema);
  if (!schema) {
    return {};
  }

  return Object.fromEntries(
    getSchemaFields(schema).map((field) => [
      field.key,
      normalizeFieldValue(field, getSettingValue(settings, field.key, source)),
    ]),
  );
}

export function inferInitialView(
  settings: Settings,
  schemaOverride?: SettingsSchema | null,
  source: SettingsValueSource = "agent_settings",
): SettingsView {
  const schema =
    schemaOverride ??
    (source === "conversation_settings"
      ? settings.conversation_settings_schema
      : settings.agent_settings_schema);
  if (!schema) {
    return "basic";
  }

  let hasMinorOverride = false;
  let hasMajorOverride = false;

  for (const field of getSchemaFields(schema)) {
    if (!isCriticalField(field)) {
      const currentValue = getSettingValue(settings, field.key, source);
      const isDifferent =
        normalizeComparableValue(
          field,
          currentValue ?? field.default ?? null,
        ) !== normalizeComparableValue(field, field.default ?? null);

      if (isDifferent) {
        if (isMinorField(field)) {
          hasMinorOverride = true;
        } else {
          hasMajorOverride = true;
        }
      }
    }
  }

  if (hasMinorOverride) return "all";
  if (hasMajorOverride) return "advanced";
  return "basic";
}

export function isSettingsFieldVisible(
  field: SettingsFieldSchema,
  values: SettingsFormValues,
): boolean {
  // "field" needs that switch on; "field=value" needs that choice selected.
  return field.depends_on.every((dependency) => {
    const [key, expected] = dependency.split("=", 2);
    return expected === undefined
      ? values[key] === true
      : values[key] === expected;
  });
}

function parseBooleanFieldValue(rawValue: string | boolean): boolean | null {
  if (typeof rawValue === "boolean") {
    return rawValue;
  }

  const normalizedValue = rawValue.trim().toLowerCase();
  if (!normalizedValue) {
    return null;
  }
  if (normalizedValue === "true") {
    return true;
  }
  if (normalizedValue === "false") {
    return false;
  }

  throw new Error(`Expected a boolean value, received: ${rawValue}`);
}

export function coerceFieldValue(
  field: SettingsFieldSchema,
  rawValue: string | boolean,
): SettingsValue {
  if (field.value_type === "boolean") {
    return parseBooleanFieldValue(rawValue);
  }

  if (field.value_type === "integer" || field.value_type === "number") {
    const stringValue = String(rawValue).trim();
    if (!stringValue) {
      return null;
    }

    const parsedValue = Number(stringValue);
    if (Number.isNaN(parsedValue)) {
      throw new Error(`Expected a numeric value, received: ${stringValue}`);
    }
    if (field.value_type === "integer" && !Number.isInteger(parsedValue)) {
      throw new Error(`Expected an integer value, received: ${stringValue}`);
    }

    // Consume SDK-supplied constraints first, fall back to the frontend's
    // hardcoded per-field metadata map (for older agent-servers that do
    // not expose these descriptors).
    const constraints = resolveNumericConstraints(field);
    if (constraints?.min != null) {
      if (constraints.exclusive_min && parsedValue <= constraints.min) {
        throw new Error(
          `${field.label} must be greater than ${constraints.min}`,
        );
      }
      if (!constraints.exclusive_min && parsedValue < constraints.min) {
        throw new Error(`${field.label} must be at least ${constraints.min}`);
      }
    }
    if (constraints?.max != null) {
      if (constraints.exclusive_max && parsedValue >= constraints.max) {
        throw new Error(`${field.label} must be less than ${constraints.max}`);
      }
      if (!constraints.exclusive_max && parsedValue > constraints.max) {
        throw new Error(`${field.label} must be at most ${constraints.max}`);
      }
    }

    return parsedValue;
  }

  if (field.value_type === "array" || field.value_type === "object") {
    const stringValue = String(rawValue).trim();
    if (!stringValue) {
      return null;
    }

    let parsedValue: unknown;
    try {
      parsedValue = JSON.parse(stringValue);
    } catch {
      throw new Error(`Invalid JSON for ${field.label}`);
    }

    if (field.value_type === "array") {
      if (!Array.isArray(parsedValue)) {
        throw new Error(`${field.label} must be a JSON array`);
      }
      return parsedValue as SettingsValue[];
    }

    if (
      parsedValue === null ||
      Array.isArray(parsedValue) ||
      typeof parsedValue !== "object"
    ) {
      throw new Error(`${field.label} must be a JSON object`);
    }

    return parsedValue as { [key: string]: SettingsValue };
  }

  const stringValue = String(rawValue);
  if (stringValue === "" && !field.secret) {
    return null;
  }

  return stringValue;
}

export function buildSdkSettingsPayload(
  schema: SettingsSchema,
  values: SettingsFormValues,
  dirty: SettingsDirtyState,
): SdkSettingsPayload {
  const payload: Record<string, unknown> = {};

  for (const field of getSchemaFields(schema)) {
    if (dirty[field.key]) {
      setDotted(payload, field.key, coerceFieldValue(field, values[field.key]));
    }
  }

  return payload as SdkSettingsPayload;
}

function isFieldVisibleInView(
  field: SettingsFieldSchema,
  view: SettingsView,
): boolean {
  return VIEW_PROMINENCES[view].has(field.prominence);
}

/**
 * Build a PATCH payload limited to the caller's current view tier.
 *
 * Contrary to the old behaviour, this no longer injects defaults for fields
 * outside the active view. The backend PATCH endpoint only applies keys that
 * are actually sent — anything omitted stays untouched. Resetting hidden keys
 * to their schema default on save caused user-edited settings to vanish the
 * moment they switched away from a view tier. The caller (e.g.
 * `SdkSectionPage`) is responsible for pre-filtering `schema` to the sections
 * its page owns; fields the page never renders never appear in `dirty`, so
 * they never reach the payload.
 *
 * This function therefore behaves identically to {@link buildSdkSettingsPayload}
 * — it exists solely to preserve the `…ForView` API surface used by callers
 * that may later need view-scoped differential handling.
 */
export function buildSdkSettingsPayloadForView(
  schema: SettingsSchema,
  values: SettingsFormValues,
  dirty: SettingsDirtyState,
  _view: SettingsView,
): SdkSettingsPayload {
  // View visibility is still consulted so that fields which the current view
  // cannot reach (and therefore cannot possibly have become dirty) never leak
  // into the payload. Crucially though, non-visible-but-previously-dirty
  // fields are NOT reset to defaults — they are simply omitted, letting the
  // backend leave them untouched.
  const payload = buildSdkSettingsPayload(schema, values, dirty) as Record<
    string,
    unknown
  >;

  // NOTE: deliberately NOT setting field.default for invisible fields here.
  // That was the bug that clobbered hidden user settings on save.

  return payload as SdkSettingsPayload;
}

/** Return sections with fields filtered for the current view tier.
 *  Specially-rendered fields are excluded from the generic list. */
export function getVisibleSettingsSections(
  schema: SettingsSchema,
  values: SettingsFormValues,
  view: SettingsView,
  excludeKeys: Set<string> = SPECIALLY_RENDERED_KEYS,
): SettingsSectionSchema[] {
  if (!isValidSettingsSchema(schema)) return [];
  return schema.sections
    .map((section) => ({
      ...section,
      fields: section.fields.filter(
        (field) =>
          !excludeKeys.has(field.key) &&
          isFieldVisibleInView(field, view) &&
          isSettingsFieldVisible(field, values),
      ),
    }))
    .filter((section) => section.fields.length > 0);
}

/** Whether the schema has any "critical" prominence fields (the basic tier). */
export function hasCriticalSettings(schema: SettingsSchema | null): boolean {
  if (!schema) return false;
  return getSchemaFields(schema).some((f) => f.prominence === "critical");
}

/** Whether the schema has any fields visible in the "advanced" tier. */
export function hasAdvancedSettings(schema: SettingsSchema | null): boolean {
  if (!schema) return false;
  return getSchemaFields(schema).some((f) => f.prominence === "major");
}

/** Whether the schema has any "minor" prominence fields. */
export function hasMinorSettings(schema: SettingsSchema | null): boolean {
  if (!schema) return false;
  return getSchemaFields(schema).some((f) => f.prominence === "minor");
}
