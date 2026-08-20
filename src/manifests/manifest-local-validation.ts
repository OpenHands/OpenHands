/**
 * The form model, and the host's own check of what the user typed into it.
 *
 * Local validation is a convenience, not the authority: it is instant and needs
 * no round trip, so it catches empty required fields and obvious typos.
 * Deployment-specific questions ("is a one-minute schedule allowed here?")
 * belong to preflight, which is authoritative.
 *
 * Errors are returned as codes rather than sentences so the host can render
 * them through its own translations; only manifest-authored copy is literal.
 */

import type {
  DeploymentCapabilities,
  SetupBlock,
  SetupFieldOption,
  SetupFormField,
  SetupFormFields,
  SetupFormValue,
  SetupFormValues,
} from "./types";

/** Characters that would break out of an expression string literal. */
const UNSAFE_EXPRESSION_LITERAL_PATTERN = /["'\\]/;

export type SetupFieldError =
  | { code: "required" }
  | { code: "minLength"; length: number }
  | { code: "maxLength"; length: number }
  | { code: "invalidOption" }
  | { code: "unsafeExpressionLiteral" };

export type SetupFieldErrors = Record<string, SetupFieldError>;

/** Field constraints supplied by the deployment rather than by the manifest. */
export interface SetupFieldOverride {
  options?: SetupFieldOption[];
}

export type SetupFieldOverrides = Record<string, SetupFieldOverride>;

/**
 * Every input the form declares, keyed by name, whichever half it is in.
 *
 * Trigger inputs come first so the user is asked when it runs before what it
 * runs on, and so every derived view of the form keeps that order. Admission
 * rejects a name declared in both halves, so the merge cannot lose a field.
 */
export function collectFields(setup: SetupBlock): SetupFormFields {
  const triggers = Object.values(setup.form.triggers ?? {});
  return Object.assign({}, ...triggers, setup.form.args) as SetupFormFields;
}

/**
 * Feed deployment values into form field constraints, so the form offers only
 * what the deployment accepts.
 *
 * A `timezone` field is the case that needs it: a manifest declares no options
 * for one, because the accepted zones belong to the deployment. The cron
 * interval floor is deliberately left to preflight — the deployment owns that
 * limit and states it in its own words, and a local approximation would
 * pre-empt the authoritative message with a worse one.
 */
export function resolveFieldOverrides(
  setup: SetupBlock,
  capabilities: DeploymentCapabilities | null,
): SetupFieldOverrides {
  const timezones = capabilities?.triggers?.cron?.timezones;
  if (!timezones?.length) return {};

  const options = timezones.map((zone) => ({ value: zone, label: zone }));
  return Object.fromEntries(
    Object.entries(collectFields(setup))
      .filter(([, field]) => field.type === "timezone")
      .map(([name]) => [name, { options }]),
  );
}

/** The options a field offers, after deployment constraints are applied. */
export function getFieldOptions(
  name: string,
  field: SetupFormField,
  overrides: SetupFieldOverrides = {},
): SetupFieldOption[] {
  return overrides[name]?.options ?? field.options ?? [];
}

/**
 * Every value a field holds, whether it collects one or many.
 *
 * Reading both shapes as a list is what keeps validation, interpolation and
 * the payload mapping from branching on `multiple` at every use.
 */
export function fieldValues(value: SetupFormValue | undefined): string[] {
  if (Array.isArray(value)) return value.filter((item) => item.trim() !== "");
  return value && value.trim() !== "" ? [value] : [];
}

/** The single value a field holds, or "" for a field collecting several. */
export function fieldText(value: SetupFormValue | undefined): string {
  return typeof value === "string" ? value : "";
}

/** Initial form state: every declared field, seeded with its declared default. */
export function getInitialFormValues(setup: SetupBlock): SetupFormValues {
  return Object.fromEntries(
    Object.entries(collectFields(setup)).map(([name, field]) => [
      name,
      // A field collecting several values starts empty rather than holding one
      // blank entry, so "required" means "add one" rather than "fill this in".
      field.multiple ? [] : (field.default ?? ""),
    ]),
  );
}

function validateField(
  name: string,
  field: SetupFormField,
  rawValue: SetupFormValue | undefined,
  overrides: SetupFieldOverrides,
): SetupFieldError | null {
  const entered = fieldValues(rawValue);

  if (entered.length === 0) {
    return field.required ? { code: "required" } : null;
  }

  // Every entry of a multi-value field answers the same field, so each is held
  // to the same rules and the first failure is the one reported.
  const failures = entered
    .map((item) => validateValue(name, field, item.trim(), overrides))
    .filter((error): error is SetupFieldError => error !== null);
  return failures[0] ?? null;
}

function validateValue(
  name: string,
  field: SetupFormField,
  value: string,
  overrides: SetupFieldOverrides,
): SetupFieldError | null {
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

  // Any field offering a closed set of values, whether the manifest declared it
  // or the deployment supplied it, must be answered from that set.
  const options = getFieldOptions(name, field, overrides);
  if (options.length > 0 && !options.some((option) => option.value === value)) {
    return { code: "invalidOption" };
  }

  return null;
}

/** Check every declared field. Returns only the fields that failed. */
export function validateFormValues(
  setup: SetupBlock,
  values: SetupFormValues,
  overrides: SetupFieldOverrides = {},
): SetupFieldErrors {
  return Object.entries(collectFields(setup)).reduce<SetupFieldErrors>(
    (errors, [name, field]) => {
      const error = validateField(name, field, values[name], overrides);
      return error ? { ...errors, [name]: error } : errors;
    },
    {},
  );
}
