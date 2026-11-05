import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { PluginSpecPreflightResponse } from "#/api/automation-service/automation-service.api";

/**
 * Manifest-aware additions to the automation launch flow (OpenHands#18793):
 * mapping a preflight response (OpenHands/automation#630) onto the free-text
 * parameter inputs the modal already renders, deriving the template used for
 * install dedupe, and building the trigger passed to the create request.
 */

/** One schema property rendered as a typed input. */
export interface SchemaFormField {
  key: string;
  type: "string" | "number" | "integer" | "boolean";
  label: string;
  helpText?: string;
  required: boolean;
  defaultValue?: unknown;
  /**
   * Display strings for enum options (raw values coerced via `String`). Used
   * for the `<option>` label and `value` attribute. Pair with `enumValues` to
   * recover the original typed value on selection.
   */
  options?: string[];
  /**
   * Raw (typed) enum values in the same order as `options`, preserved so a
   * numeric or boolean enum isn't coerced to a string on submit. When absent,
   * `options` strings are passed through as-is.
   */
  enumValues?: unknown[];
  /**
   * UI hint for string fields. `"textarea"` renders a multi-line input; useful
   * for values that are naturally multi-line (feed lists, topics). Derived from
   * the JSON schema `format` convention.
   */
  widget?: "textarea";
}

type ConfigSchemaProperty = Record<string, unknown>;

function propertyType(property: ConfigSchemaProperty): SchemaFormField["type"] {
  if (property.type === "integer") return "integer";
  if (property.type === "number") return "number";
  if (property.type === "boolean") return "boolean";
  return "string";
}

/**
 * Map a manifest's `config_schema` (flat object properties) onto typed form
 * fields. Properties absent from the URL-pre-encoded parameters fall back to
 * the schema's `default`; keys the schema doesn't declare stay free-text
 * (they are passed through unchanged to the create request).
 */
export function configSchemaToFormFields(
  preflight: PluginSpecPreflightResponse,
): SchemaFormField[] {
  const properties = preflight.manifest.config_schema?.properties;
  if (!properties) return [];

  const required = new Set(preflight.manifest.config_schema?.required ?? []);

  return Object.entries(properties).map(([key, property]) => {
    const field: SchemaFormField = {
      key,
      type: propertyType(property),
      label: (property.title as string | undefined) ?? key,
      helpText: property.description as string | undefined,
      required: required.has(key),
      defaultValue: property.default,
    };
    if (Array.isArray(property.enum) && property.enum.length > 0) {
      field.options = property.enum.map((option) => String(option));
      field.enumValues = property.enum;
    }
    if (
      property.format === "textarea" ||
      property["ui:widget"] === "textarea"
    ) {
      field.widget = "textarea";
    }
    return field;
  });
}

/**
 * Apply a schema to URL-pre-encoded parameters: fields the schema declares but
 * the URL omitted get their schema default pre-filled (so the form shows the
 * full expected shape). Fields with no default are left undefined — the backend
 * applies its own defaults, and the create request omits them, so a required
 * field that the user never touched surfaces as a 422 rather than silently
 * installing a broken automation with an empty-string value.
 */
export function applySchemaDefaults(
  parameters: Record<string, unknown>,
  fields: SchemaFormField[],
): Record<string, unknown> {
  const merged: Record<string, unknown> = { ...parameters };
  for (const field of fields) {
    if (merged[field.key] !== undefined) continue;
    if (field.defaultValue !== undefined) {
      merged[field.key] = field.defaultValue;
    }
    // No default: leave undefined. The field renders empty, the create
    // request omits it, and a required field surfaces as a 422.
  }
  return merged;
}

/**
 * Strip fields the user left empty (undefined, empty string, null) so the
 * create request only carries values the user actually set. Prevents an
 * empty string from passing a required check on the backend (key present,
 * value falsy) while also avoiding 422s for optional numeric fields sent as
 * 0 when the user never touched them.
 */
export function stripEmptyConfigValues(
  config: Record<string, unknown>,
): Record<string, unknown> {
  const cleaned: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(config)) {
    if (value === undefined || value === null || value === "") continue;
    cleaned[key] = value;
  }
  return cleaned;
}

/**
 * Stable template id for install dedupe: repeated badge clicks return the
 * existing automation (HTTP 200) instead of creating a duplicate. Uses
 * positional slots rather than a `/`-joined path so values containing `/`
 * (common for repo_path) can't collide with a different source/ref/path combo.
 */
export function pluginSpecTemplateId(plugin: PluginSpec): string {
  return `plugin-spec:${plugin.source}|${plugin.repo_path ?? ""}|${plugin.ref ?? ""}`;
}

/**
 * Template version for dedupe: the resolved ref (falling back to the raw ref,
 * then "unresolved"). The backend requires a non-empty version.
 */
export function pluginSpecTemplateVersion(
  resolvedRef: string | null | undefined,
  plugin: PluginSpec,
): string {
  return resolvedRef || plugin.ref || "unresolved";
}

/**
 * The cron trigger rendered from the preflight's `suggested_trigger`, as the
 * user (possibly) edited it. Only cron suggestions are editable here today;
 * event suggestions pass through unchanged for the service to validate.
 */
export function buildCronTrigger(
  suggested: PluginSpecPreflightResponse["suggested_trigger"],
  schedule: string,
  timezone: string,
): { type: "cron"; schedule: string; timezone: string } | null {
  if (!suggested) return null;
  if (suggested.type !== "cron") return null;
  return { type: "cron", schedule, timezone };
}
