/**
 * Translate service-reported errors back to the form field that produced them.
 *
 * The host validates and submits the *derived payload*, so errors come back
 * addressed by payload path (`trigger.schedule`) rather than by form field
 * (`schedule`). The reverse lookup is derived from the same builder that made
 * the payload; the mapping is lossy by nature, since one payload value can be
 * built from several fields.
 */

import type { SetupRequestBody } from "./types";

/** An error the service reported, addressed by payload path where it gave one. */
export interface ManifestServiceError {
  path: string;
  message: string;
}

export interface MappedManifestErrors {
  /** Keyed by form field name. */
  fieldErrors: Record<string, string>;
  /** Errors with no field to attach to; shown against the form as a whole. */
  formErrors: string[];
}

const EMPTY_MAPPED_ERRORS: MappedManifestErrors = {
  fieldErrors: {},
  formErrors: [],
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Turn a service `loc` array into a path into the payload actually sent.
 *
 * A validation framework may address a value through segments that are not keys
 * of the request body — a discriminated union reports
 * `["body","trigger","cron","schedule"]` for a body whose trigger has no `cron`
 * key. Walking the payload and skipping segments that do not resolve reduces
 * that to `trigger.schedule` without knowing anything about the payload's
 * meaning.
 */
export function locToPayloadPath(
  loc: readonly (string | number)[],
  payload: unknown,
): string {
  const segments = loc[0] === "body" ? loc.slice(1) : loc;
  const parts: string[] = [];
  let current: unknown = payload;

  segments.forEach((segment) => {
    if (typeof segment === "number") {
      if (Array.isArray(current) && parts.length > 0) {
        parts[parts.length - 1] += `[${segment}]`;
        current = current[segment];
      }
      return;
    }
    if (isRecord(current) && segment in current) {
      parts.push(segment);
      current = current[segment];
    }
  });

  return parts.join(".");
}

/**
 * Read errors out of a service response body, whichever of the two shapes it
 * uses: a preflight result keyed by field, or a validation failure keyed by
 * `loc`.
 */
export function normalizeServiceErrors(
  body: unknown,
  payload: SetupRequestBody | null,
): ManifestServiceError[] {
  if (!isRecord(body)) return [];

  if (Array.isArray(body.errors)) {
    return body.errors.flatMap((entry) => {
      if (!isRecord(entry) || typeof entry.message !== "string") return [];
      return [
        {
          path: typeof entry.field === "string" ? entry.field : "",
          message: entry.message,
        },
      ];
    });
  }

  const { detail } = body;
  if (typeof detail === "string") {
    return [{ path: "", message: detail }];
  }
  if (Array.isArray(detail)) {
    return detail.flatMap((entry) => {
      if (!isRecord(entry) || typeof entry.msg !== "string") return [];
      const loc = Array.isArray(entry.loc)
        ? (entry.loc as (string | number)[])
        : [];
      return [{ path: locToPayloadPath(loc, payload), message: entry.msg }];
    });
  }

  return [];
}

/** Apply the derived payload-path map, falling back to a form-level error. */
export function mapServiceErrors(
  errors: readonly ManifestServiceError[],
  errorMap: Record<string, string[]>,
): MappedManifestErrors {
  if (errors.length === 0) return EMPTY_MAPPED_ERRORS;

  const fieldErrors: Record<string, string> = {};
  const formErrors: string[] = [];

  errors.forEach(({ path, message }) => {
    const fields = path ? errorMap[path] : undefined;
    if (!fields?.length) {
      formErrors.push(message);
      return;
    }
    fields.forEach((field) => {
      // Keep the first error reported against a field; later ones would
      // silently replace a message the user has not read yet.
      if (!(field in fieldErrors)) fieldErrors[field] = message;
    });
  });

  return { fieldErrors, formErrors };
}
