/**
 * Placeholder interpolation for setup copy and request-body strings.
 *
 * A setup block declares what the user sees and the two request strings the
 * host cannot derive as templates over a small set of namespaces. Interpolation
 * is plain substitution — there is no expression language, so a setup block
 * cannot express behavior here.
 */

import type { SetupEntry, SetupFormValues } from "./types";

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
  // Missing values render as blank; callers that care show their own fallback.
  return "";
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
