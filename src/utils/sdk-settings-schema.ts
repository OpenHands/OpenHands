import {
  SettingProminence,
  Settings,
  SettingsFieldSchema,
  SettingsSchema,
  SettingsSectionSchema,
  SettingsValue,
} from "#/types/settings";
import { getSettingsFieldConstraints } from "#/utils/sdk-settings-field-metadata";
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

export function getConversationSettingValue(
  settings: Settings,
  key: string,
): SettingsValue {
  return getSettingValue(settings, key, "conversation_settings");
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

/** Determine which view tier to default to based on whether the user has
 *  overridden any non-critical settings. */
export function hasAdvancedSettingsOverrides(settings: Settings): boolean {
  return inferInitialView(settings) !== "basic";
}

export function isSettingsFieldVisible(
  field: SettingsFieldSchema,
  values: SettingsFormValues,
): boolean {
  return field.depends_on.every((dependency) => values[dependency] === true);
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

/** Fields whose value is a URL, by naming convention. Decides both the
 *  `type="url"` input rendering in `SchemaField` and the save-time format
 *  check in `coerceFieldValue`, so the two can never disagree. */
export function isUrlField(field: SettingsFieldSchema): boolean {
  return field.key.endsWith("url") || field.key.endsWith("_url");
}

/** Fields whose value is an API key, by naming convention. Same
 *  render/enforce pairing as {@link isUrlField}. */
export function isApiKeyField(field: SettingsFieldSchema): boolean {
  return field.key.endsWith("api_key");
}

/** Minimum length for a non-blank API key: a single character (the "-" from
 *  #15774) is never a usable credential. Blank stays valid — it means
 *  "unset" for these optional fields. */
export const MIN_API_KEY_LENGTH = 2;

/** A URL we are willing to send to a provider: parseable, and http(s)
 *  rather than `file:`, `javascript:` or a bare host that parses as its own
 *  scheme (`localhost:8000` yields `protocol === "localhost:"`). Matches
 *  the check MCP server URLs already get in `mcp-server-form.tsx`. */
export function isValidSettingsUrl(value: string): boolean {
  try {
    const { protocol } = new URL(value);
    return protocol === "http:" || protocol === "https:";
  } catch {
    return false;
  }
}

/** A non-blank API key is at least {@link MIN_API_KEY_LENGTH} characters
 *  once trimmed. Callers gate on blank separately, same as
 *  {@link isValidSettingsUrl}. */
export function isValidApiKey(value: string): boolean {
  return value.trim().length >= MIN_API_KEY_LENGTH;
}

/** Codes identifying which format rule a field value violates. Renderers
 *  map each code to a translated inline message; `coerceFieldValue` maps
 *  them to the save-blocking errors below. */
export type SettingsFieldFormatError = "url" | "api_key";

/**
 * Single source of truth for per-field format checks, shared by the
 * save-time coercion and the inline error rendering so they never
 * disagree. Blank values return null — optional fields use blank to mean
 * "unset" — and fields without a format rule return null.
 */
export function getSettingsFieldFormatError(
  field: SettingsFieldSchema,
  rawValue: string | boolean,
): SettingsFieldFormatError | null {
  const trimmedValue = String(rawValue).trim();
  if (!trimmedValue) {
    return null;
  }
  if (isUrlField(field) && !isValidSettingsUrl(trimmedValue)) {
    return "url";
  }
  if (isApiKeyField(field) && !isValidApiKey(trimmedValue)) {
    return "api_key";
  }
  return null;
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

    const constraints = getSettingsFieldConstraints(field.key);
    if (constraints?.min != null && parsedValue < constraints.min) {
      throw new Error(`${field.label} must be at least ${constraints.min}`);
    }
    if (constraints?.max != null && parsedValue > constraints.max) {
      throw new Error(`${field.label} must be at most ${constraints.max}`);
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

  // Reject malformed values here rather than saving them: the field stays
  // optional when blank, but a value like "." is otherwise accepted all the
  // way to a green "saved" toast and only resurfaces later as an opaque
  // provider error.
  const formatError = getSettingsFieldFormatError(field, stringValue);
  if (formatError === "url") {
    throw new Error(`${field.label} must use http:// or https://`);
  }
  if (formatError === "api_key") {
    throw new Error(
      `${field.label} must be at least ${MIN_API_KEY_LENGTH} characters`,
    );
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

export function buildSdkSettingsPayloadForView(
  schema: SettingsSchema,
  values: SettingsFormValues,
  dirty: SettingsDirtyState,
  view: SettingsView,
): SdkSettingsPayload {
  const payload = buildSdkSettingsPayload(schema, values, dirty) as Record<
    string,
    unknown
  >;

  for (const field of getSchemaFields(schema)) {
    if (!isFieldVisibleInView(field, view)) {
      setDotted(payload, field.key, field.default ?? null);
    }
  }

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
