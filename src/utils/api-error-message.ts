import axios from "axios";

/**
 * Extract a human-readable message from a failed API call.
 *
 * Handles both transports the app uses: local agent-server calls that
 * throw an `AxiosError` (body under `error.response.data`) and cloud
 * calls through the shared TypeScript client that throw an `HttpError`
 * (parsed body directly under `error.response`). Prefers the
 * server-provided `message`/`detail` fields, then the `Error` message,
 * then `fallback`.
 */
export function getApiErrorMessage(error: unknown, fallback: string): string {
  const body = axios.isAxiosError(error)
    ? error.response?.data
    : error instanceof Error && "response" in error
      ? (error as { response?: unknown }).response
      : undefined;

  if (body && typeof body === "object") {
    const { message, detail } = body as {
      message?: unknown;
      detail?: unknown;
    };
    if (typeof message === "string" && message) return message;
    if (typeof detail === "string" && detail) return detail;
  }

  if (error instanceof Error && error.message) return error.message;
  return fallback;
}
