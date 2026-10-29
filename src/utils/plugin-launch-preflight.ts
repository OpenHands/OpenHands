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
  type: "string" | "number" | "boolean";
  label: string;
  helpText?: string;
  required: boolean;
  defaultValue?: unknown;
  options?: string[];
  /**
   * UI hint for string fields. `"textarea"` renders a multi-line input; useful
   * for values that are naturally multi-line (feed lists, topics). Derived from
   * the JSON schema `format` convention.
   */
  widget?: "textarea";
}

type ConfigSchemaProperty = Record<string, unknown>;

function propertyType(property: ConfigSchemaProperty): SchemaFormField["type"] {
  return property.type === "number" || property.type === "integer"
    ? "number"
    : property.type === "boolean"
      ? "boolean"
      : "string";
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
 * the URL omitted get their schema default (or a type-appropriate empty value)
 * pre-filled, so the form shows the full expected shape rather than only what
 * the badge author encoded.
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
      continue;
    }
    switch (field.type) {
      case "number":
        merged[field.key] = 0;
        break;
      case "boolean":
        merged[field.key] = false;
        break;
      default:
        merged[field.key] = field.options?.[0] ?? "";
    }
  }
  return merged;
}

/**
 * Stable template id for install dedupe: repeated badge clicks return the
 * existing automation (HTTP 200) instead of creating a duplicate.
 */
export function pluginSpecTemplateId(plugin: PluginSpec): string {
  const path = [plugin.source, plugin.repo_path, plugin.ref]
    .filter((part): part is string => Boolean(part))
    .join("/");
  return `plugin-spec:${path}`;
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
