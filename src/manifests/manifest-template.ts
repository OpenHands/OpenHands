/**
 * Placeholder interpolation for setup copy and request-body strings.
 *
 * A setup block declares what the user sees and the two request strings the
 * host cannot derive as templates over a small set of namespaces. Interpolation
 * is plain substitution — there is no expression language, so a setup block
 * cannot express behavior here.
 */

import type { SetupEntry, SetupFormValues, SetupPayloadValue } from "./types";

const PLACEHOLDER_PATTERN = /\{\{([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)\}\}/g;

export interface SetupScope {
  form?: SetupFormValues;
  /** The catalog entry the setup block belongs to. */
  automation?: SetupEntry;
}

/** Walk a dotted path through plain objects. */
export function getByPath(source: unknown, path: string): unknown {
  return path.split(".").reduce<unknown>((current, segment) => {
    if (typeof current !== "object" || current === null) return undefined;
    return (current as Record<string, unknown>)[segment];
  }, source);
}

function toText(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  // A multi-value field inside a sentence reads as a list of names.
  if (Array.isArray(value)) return value.map(toText).filter(Boolean).join(", ");
  // Missing values render as blank; callers that care show their own fallback.
  return "";
}

/** A template that is exactly one placeholder, or null if it is not. */
function wholePlaceholderPath(template: string): string | null {
  const match = /^\{\{([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)\}\}$/.exec(template);
  return match ? match[1] : null;
}

function isPlainPayloadValue(value: unknown): value is SetupPayloadValue {
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return true;
  }
  if (Array.isArray(value)) return value.every(isPlainPayloadValue);
  if (typeof value !== "object") return false;
  if (Object.getPrototypeOf(value) !== Object.prototype) return false;
  return Object.values(value).every(isPlainPayloadValue);
}

/**
 * Substitute placeholders, keeping the resolved value's own type when the
 * template is nothing but that placeholder.
 *
 * This is what lets a request body state `"repos": "{{form.repositories}}"` and
 * get an array. Inside a sentence the same placeholder still reads as text,
 * because there is nowhere for a list to go in a string.
 *
 * Plain JSON form values are kept whole. A placeholder naming anything else -
 * for example `{{automation.setup}}`, or a browser `File` selected by the user -
 * reads as text like it does inside a sentence, so a manifest cannot state one
 * value and put its own object graph into the request body.
 */
export function interpolateValue(
  template: string,
  scope: SetupScope,
): SetupPayloadValue {
  const path = wholePlaceholderPath(template);
  if (path === null) return interpolateText(template, scope);
  const resolved = getByPath(scope, path);
  return isPlainPayloadValue(resolved) ? resolved : toText(resolved);
}

/** Substitute placeholders inside a template string. */
export function interpolateText(template: string, scope: SetupScope): string {
  return template.replace(PLACEHOLDER_PATTERN, (_match, path: string) =>
    toText(getByPath(scope, path)),
  );
}

/** Substitute placeholders from a flat name → value record. */
export function interpolateValues(
  template: string,
  values: Record<string, string | number>,
): string {
  return template.replace(PLACEHOLDER_PATTERN, (_match, name: string) =>
    toText(values[name]),
  );
}
