import axios from "axios";

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
 * Extract a human-readable message from a failed API call. Prefers the
 * server-provided `message`/`exception`/`detail` fields, then the `Error`
 * message, then `fallback`. The Agent Server answers an unhandled error with
 * a fixed `detail: "Internal Server Error"` and the actual reason under
 * `exception`, so that one wins over `detail`.
 */
export function getApiErrorMessage(error: unknown, fallback: string): string {
  const body = getApiErrorBody(error);

  if (body && typeof body === "object") {
    const { message, exception, detail } = body as {
      message?: unknown;
      exception?: unknown;
      detail?: unknown;
    };
    if (typeof message === "string" && message) return message;
    if (typeof exception === "string" && exception) return exception;
    if (typeof detail === "string" && detail) return detail;
  }

  if (error instanceof Error && error.message) return error.message;
  return fallback;
}
