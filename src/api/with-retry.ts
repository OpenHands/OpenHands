/**
 * The HTTP status a failed call answered with, from either transport the app
 * uses: the shared TypeScript client's `HttpError` (`status`) or Axios
 * (`response.status`). `undefined` when the server never answered, e.g. a
 * network error or a timeout.
 */
function getHttpStatus(error: unknown): number | undefined {
  if (typeof error !== "object" || error === null) return undefined;
  const { response, status } = error as {
    response?: { status?: unknown };
    status?: unknown;
  };
  if (typeof response?.status === "number") return response.status;
  if (typeof status === "number") return status;
  return undefined;
}

/**
 * Whether a failed call may succeed if sent again. A 4xx answer is the
 * server rejecting the request itself (validation, auth, not found), so the
 * same request gets the same answer; only 408 (timeout) and 429 (rate limit)
 * among them are worth another try. Server errors (5xx) and failures with no
 * answer at all can be transient.
 */
export function isRetryableError(error: unknown): boolean {
  const status = getHttpStatus(error);
  if (status === undefined) return true;
  if (status === 408 || status === 429) return true;
  return status < 400 || status >= 500;
}

/**
 * Retry helper for API calls with exponential backoff. Errors that
 * {@link isRetryableError} rejects are thrown on the first attempt.
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  baseDelayMs: number = 500,
): Promise<T> {
  for (let attempt = 0; attempt < maxRetries; attempt += 1) {
    try {
      return await fn();
    } catch (error) {
      if (attempt >= maxRetries - 1 || !isRetryableError(error)) {
        throw error;
      }

      const delay = baseDelayMs * 2 ** attempt;

      await new Promise<void>((resolve) => {
        setTimeout(resolve, delay);
      });
    }
  }

  throw new Error("Retry attempts exhausted");
}
