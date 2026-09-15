/**
 * Retry helper for API calls with exponential backoff.
 *
 * `shouldRetry` decides per failure whether another attempt is worthwhile
 * (e.g. retry a transient 5xx but fail fast on a 4xx). It defaults to
 * retrying every failure, preserving the previous behavior.
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  baseDelayMs: number = 500,
  shouldRetry: (error: unknown) => boolean = () => true,
): Promise<T> {
  for (let attempt = 0; attempt < maxRetries; attempt += 1) {
    try {
      return await fn();
    } catch (error) {
      if (attempt >= maxRetries - 1 || !shouldRetry(error)) {
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
