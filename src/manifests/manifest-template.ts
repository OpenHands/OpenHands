/**
 * Placeholder interpolation for manifest copy and request bodies.
 *
 * A manifest declares what the user sees and what the host sends as templates
 * over a small set of namespaces. Interpolation is plain substitution — there
 * is no expression language, so a manifest cannot express behavior here.
 */

import type {
  ExtensionManifest,
  ManifestFormValues,
  ManifestPayloadValue,
  ManifestRequestBody,
} from "./types";

const PLACEHOLDER_PATTERN = /\{\{([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)\}\}/g;
const WHOLE_PLACEHOLDER_PATTERN =
  /^\{\{([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)\}\}$/;

export interface ManifestScope {
  form?: ManifestFormValues;
  manifest?: ExtensionManifest;
  /** The action's response, e.g. the created resource or the new conversation. */
  response?: Record<string, unknown>;
  /** The capability discovery result plus the host's `supported` verdict. */
  capabilities?: Record<string, unknown>;
  /** Carries `payload`, so preflight can validate exactly what will be sent. */
  submit?: { payload?: ManifestRequestBody };
}

/** Walk a dotted path through plain objects. Used for placeholders and bindings. */
export function getByPath(source: unknown, path: string): unknown {
  return path.split(".").reduce<unknown>((current, segment) => {
    if (typeof current !== "object" || current === null) return undefined;
    return (current as Record<string, unknown>)[segment];
  }, source);
}

function resolvePlaceholder(path: string, scope: ManifestScope): unknown {
  return getByPath(scope, path);
}

function toText(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  // Missing values render as blank; callers that care show their own fallback.
  return "";
}

/** Substitute placeholders inside a copy string. */
export function interpolateText(
  template: string,
  scope: ManifestScope,
): string {
  return template.replace(PLACEHOLDER_PATTERN, (_match, path: string) =>
    toText(resolvePlaceholder(path, scope)),
  );
}

/**
 * Substitute placeholders inside a request-body value.
 *
 * A string that is exactly one placeholder adopts the resolved value's own
 * type, so `"{{submit.payload}}"` sends the payload object rather than a
 * stringified copy of it.
 */
export function interpolateValue(
  value: ManifestPayloadValue,
  scope: ManifestScope,
): ManifestPayloadValue {
  if (typeof value === "string") {
    const whole = value.match(WHOLE_PLACEHOLDER_PATTERN);
    if (whole) {
      const resolved = resolvePlaceholder(whole[1], scope);
      if (typeof resolved === "object" && resolved !== null) {
        return resolved as ManifestPayloadValue;
      }
      return toText(resolved);
    }
    return interpolateText(value, scope);
  }
  if (Array.isArray(value)) {
    return value.map((item) => interpolateValue(item, scope));
  }
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        key,
        interpolateValue(item, scope),
      ]),
    );
  }
  return value;
}

/** Map a declared request body into the body that will actually be sent. */
export function buildRequestBody(
  body: ManifestRequestBody,
  scope: ManifestScope,
): ManifestRequestBody {
  return interpolateValue(body, scope) as ManifestRequestBody;
}
