import axios from "axios";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { isSdkHttpError } from "#/utils/sdk-http-error";

/**
 * Extract the parsed response body from a failed API call.
 *
 * Handles both transports the app uses: local agent-server calls that
 * throw an `AxiosError` (body under `error.response.data`) and cloud
 * calls through the shared TypeScript client that throw an `HttpError`
 * (parsed body directly under `error.response`).
 */
export function getApiErrorBody(error: unknown): unknown {
  if (axios.isAxiosError(error)) return error.response?.data;
  if (error instanceof Error && "response" in error) {
    return (error as { response?: unknown }).response;
  }
  return undefined;
}

/** The last string segment of a validation error's `loc`, e.g. the field. */
function getValidationField(loc: unknown): string | null {
  if (!Array.isArray(loc)) return null;
  const field = [...loc].reverse().find((part) => typeof part === "string");
  return typeof field === "string" && field ? field : null;
}

/**
 * Join the `msg` fields of a FastAPI/Pydantic validation `detail` array
 * (`[{ loc, msg, type }, ...]`), or return null when there are none. With
 * several errors each message is prefixed with its field so they can be
 * told apart.
 */
function getValidationDetailMessage(detail: unknown): string | null {
  if (!Array.isArray(detail)) return null;
  const entries = detail
    .filter(
      (item): item is { loc?: unknown; msg: string } =>
        !!item &&
        typeof item === "object" &&
        typeof (item as { msg?: unknown }).msg === "string" &&
        (item as { msg: string }).msg !== "",
    )
    .map(({ loc, msg }) => ({ field: getValidationField(loc), msg }));
  if (entries.length === 0) return null;
  if (entries.length === 1) return entries[0].msg;
  return entries
    .map(({ field, msg }) => (field ? `${field}: ${msg}` : msg))
    .join("; ");
}

/**
 * Extract a human-readable message from a failed API call. Prefers the
 * server-provided `message`/`detail`/`error` fields (including a FastAPI
 * validation `detail` array), then the `Error` message, then `fallback`.
 * The shared client's `HttpError` message is the raw transport text
 * (`HTTP request failed (status): {json}`), so it is never shown: an
 * `HttpError` without a usable body yields `fallback`.
 */
export function getApiErrorMessage(error: unknown, fallback: string): string {
  const body = getApiErrorBody(error);

  if (body && typeof body === "object") {
    const {
      message,
      detail,
      error: bodyError,
    } = body as {
      message?: unknown;
      detail?: unknown;
      error?: unknown;
    };
    if (typeof message === "string" && message) return message;
    if (typeof detail === "string" && detail) return detail;
    const validationMessage = getValidationDetailMessage(detail);
    if (validationMessage) return validationMessage;
    if (typeof bodyError === "string" && bodyError) return bodyError;
  }

  if (isSdkHttpError(error)) return fallback;
  if (error instanceof Error && error.message) return error.message;
  return fallback;
}

/**
 * Like {@link getApiErrorMessage}, but a request that never got an answer
 * (a network failure or timeout: no response body and no `HttpError`) keeps
 * the shared "Disconnected (...)" wording instead of the client's raw
 * `Request failed: ...` text.
 */
export function getApiOrConnectionErrorMessage(
  error: unknown,
  fallback: string,
): string {
  if (getApiErrorBody(error) || isSdkHttpError(error)) {
    return getApiErrorMessage(error, fallback);
  }
  return retrieveAxiosErrorMessage(error) || fallback;
}
