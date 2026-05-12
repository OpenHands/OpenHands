/**
 * Pull a user-facing message off an unknown error, preferring FastAPI's
 * `detail` field, then a generic `message` body field, then `Error.message`,
 * and finally the supplied fallback. Tolerant of non-Axios errors so callers
 * don't need to narrow the type before calling.
 */
export function extractErrorMessage(err: unknown, fallback: string): string {
  if (typeof err === "object" && err !== null) {
    const withResponse = err as {
      response?: { data?: { detail?: unknown; message?: unknown } };
      message?: string;
    };
    const detail = withResponse.response?.data?.detail;
    if (typeof detail === "string") return detail;
    const message = withResponse.response?.data?.message;
    if (typeof message === "string") return message;
    if (withResponse.message) return withResponse.message;
  }
  return fallback;
}
