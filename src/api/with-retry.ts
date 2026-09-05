/**
 * Determine whether an error is worth retrying. Client errors (4xx) and
 * certain network errors are non-retryable, since retrying them wastes
 * time and can produce misleading error messages.
 */
function isRetryableError(error: unknown): boolean {
  // Re-throw AbortError - the caller intentionally cancelled the request.
  if (error instanceof DOMException && error.name === "AbortError") {
    return false;
  }

  // Check for HTTP response status codes via common API client patterns.
  const status = getErrorStatus(error);
  if (status !== undefined) {
    // 408 Request Timeout and 429 Too Many Requests are retryable.
    // All other 4xx errors are client errors that won't change on retry.
    // 5xx errors are server errors worth retrying.
    return status === 408 || status === 429 || status >= 500;
  }

  // Network errors (no response received) are retryable.
  return true;
}

function getErrorStatus(error: unknown): number | undefined {
  if (error == null || typeof error !== "object") return undefined;
  const err = error as Record<string, unknown>;
  // Axios-style: error.response.status
  if (typeof err.response === "object" && err.response !== null) {
    const response = err.response as Record<string, unknown>;
    if (typeof response.status === "number") return response.status;
  }
  // Fetch-style: error.status (Response objects thrown directly)
  if (typeof err.status === "number") return err.status;
  return undefined;
}

/**
 * Retry helper for API calls with exponential backoff.
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
