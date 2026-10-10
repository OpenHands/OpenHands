function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isCancellationError(error: unknown): boolean {
  if (!isRecord(error)) return false;

  if (
    error.name === "AbortError" ||
    error.name === "CanceledError" ||
    error.code === "ERR_CANCELED"
  ) {
    return true;
  }

  const { cause } = error;
  return (
    isRecord(cause) &&
    (cause.name === "AbortError" ||
      cause.name === "CanceledError" ||
      cause.code === "ERR_CANCELED")
  );
}

function getErrorStatus(error: unknown): number | undefined {
  if (!isRecord(error)) return undefined;
  if (typeof error.status === "number") return error.status;

  const { response } = error;
  return isRecord(response) && typeof response.status === "number"
    ? response.status
    : undefined;
}

function isRetryableError(error: unknown): boolean {
  if (isCancellationError(error)) return false;

  const status = getErrorStatus(error);
  if (status === undefined) return true;

  return status === 408 || status === 429 || status >= 500;
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
