/**
 * Stage 5 — the host's own check of form input against the constraints a
 * manifest declares.
 *
 * This is a convenience, not the authority: it is instant and needs no round
 * trip, so it catches empty required fields and obvious typos. Deployment-
 * specific questions ("is a one-minute schedule allowed here?") belong to
 * preflight, which is authoritative.
 *
 * Errors are returned as codes rather than sentences so the host can render
 * them through its own translations; only manifest-authored copy is literal.
 */

import { getByPath } from "./manifest-template";
import type {
  ExtensionManifest,
  ManifestFieldOption,
  ManifestFormField,
  ManifestFormValues,
} from "./types";

/** Characters that would break out of an expression string literal. */
const UNSAFE_EXPRESSION_LITERAL_PATTERN = /["'\\]/;

export type ManifestFieldError =
  | { code: "required" }
  | { code: "minLength"; length: number }
  | { code: "maxLength"; length: number }
  | { code: "invalidOption" }
  | { code: "unsafeExpressionLiteral" };

export type ManifestFieldErrors = Record<string, ManifestFieldError>;

/** Field constraints supplied by the deployment rather than by the manifest. */
export interface ManifestFieldOverride {
  options?: ManifestFieldOption[];
}

export type ManifestFieldOverrides = Record<string, ManifestFieldOverride>;

function toOptions(value: unknown): ManifestFieldOption[] | null {
  if (!Array.isArray(value)) return null;
  const options = value.filter(
    (item): item is string => typeof item === "string" && item.length > 0,
  );
  return options.length > 0
    ? options.map((option) => ({ value: option, label: option }))
    : null;
}

/**
 * Feed deployment values into form field constraints, so the form offers only
 * what the deployment accepts.
 *
 * Only `options` changes what the form renders. A `minIntervalSeconds` binding
 * is deliberately left to preflight: the deployment owns that limit and states
 * it in its own words, and a local approximation would pre-empt the
 * authoritative message with a worse one.
 */
export function resolveFieldOverrides(
  manifest: ExtensionManifest,
  capabilities: Record<string, unknown> | null,
): ManifestFieldOverrides {
  const bindings = manifest.capabilities?.bindings;
  if (!bindings || !capabilities) return {};

  return bindings.reduce<ManifestFieldOverrides>((overrides, binding) => {
    if (binding.constraint !== "options") return overrides;
    const options = toOptions(getByPath(capabilities, binding.from));
    if (!options) return overrides;
    return { ...overrides, [binding.field]: { options } };
  }, {});
}

/** The options a select field offers, after deployment bindings are applied. */
export function getFieldOptions(
  field: ManifestFormField,
  overrides: ManifestFieldOverrides = {},
): ManifestFieldOption[] {
  return overrides[field.name]?.options ?? field.options ?? [];
}

/** Initial form state: every declared field, seeded with its declared default. */
export function getInitialFormValues(
  manifest: ExtensionManifest,
): ManifestFormValues {
  return Object.fromEntries(
    manifest.form.fields.map((field) => [field.name, field.default ?? ""]),
  );
}

function validateField(
  field: ManifestFormField,
  rawValue: string | undefined,
  overrides: ManifestFieldOverrides,
): ManifestFieldError | null {
  const value = (rawValue ?? "").trim();

  if (!value) {
    return field.required ? { code: "required" } : null;
  }

  const { minLength, maxLength, format } = field.constraints ?? {};
  if (minLength !== undefined && value.length < minLength) {
    return { code: "minLength", length: minLength };
  }
  if (maxLength !== undefined && value.length > maxLength) {
    return { code: "maxLength", length: maxLength };
  }
  if (
    format === "safeExpressionLiteral" &&
    UNSAFE_EXPRESSION_LITERAL_PATTERN.test(value)
  ) {
    return { code: "unsafeExpressionLiteral" };
  }

  if (field.type === "select") {
    const options = getFieldOptions(field, overrides);
    if (
      options.length > 0 &&
      !options.some((option) => option.value === value)
    ) {
      return { code: "invalidOption" };
    }
  }

  return null;
}

/** Check every declared field. Returns only the fields that failed. */
export function validateFormValues(
  manifest: ExtensionManifest,
  values: ManifestFormValues,
  overrides: ManifestFieldOverrides = {},
): ManifestFieldErrors {
  return manifest.form.fields.reduce<ManifestFieldErrors>((errors, field) => {
    const error = validateField(field, values[field.name], overrides);
    return error ? { ...errors, [field.name]: error } : errors;
  }, {});
}
