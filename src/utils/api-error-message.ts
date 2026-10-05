import axios from "axios";
import { isSdkHttpError } from "#/api/agent-server-compatibility";

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

/**
 * Join the `msg` fields of a FastAPI/Pydantic validation `detail` array
 * (`[{ loc, msg, type }, ...]`), or return null when there are none.
 */
function getValidationDetailMessage(detail: unknown): string | null {
  if (!Array.isArray(detail)) return null;
  const messages = detail
    .map((item) =>
      item && typeof item === "object"
        ? (item as { msg?: unknown }).msg
        : undefined,
    )
    .filter((msg): msg is string => typeof msg === "string" && msg !== "");
  return messages.length > 0 ? messages.join("; ") : null;
}

/**
 * Extract a human-readable message from a failed API call. Prefers the
 * server-provided `message`/`detail` fields (including a FastAPI validation
 * `detail` array), then the `Error` message, then `fallback`. The shared
 * client's `HttpError` message is the raw transport text
 * (`HTTP request failed (status): {json}`), so it is never shown: an
 * `HttpError` without a usable body yields `fallback`.
 */
export function getApiErrorMessage(error: unknown, fallback: string): string {
  const body = getApiErrorBody(error);

  if (body && typeof body === "object") {
    const { message, detail } = body as {
      message?: unknown;
      detail?: unknown;
    };
    if (typeof message === "string" && message) return message;
    if (typeof detail === "string" && detail) return detail;
    const validationMessage = getValidationDetailMessage(detail);
    if (validationMessage) return validationMessage;
  }

  if (isSdkHttpError(error)) return fallback;
  if (error instanceof Error && error.message) return error.message;
  return fallback;
}
