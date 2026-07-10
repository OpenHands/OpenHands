import { AxiosError } from "axios";

const INITIAL_RETRY_DELAY_MS = 1000;
const MAX_BASE_RETRY_DELAY_MS = 60_000;
const MAX_JITTER_MS = 5000;
const RETRY_AFTER_HEADER = "retry-after";

function getHeaderValue(headers: unknown, name: string): string | undefined {
  if (!headers || typeof headers !== "object") {
    return undefined;
  }

  if (
    "get" in headers &&
    typeof (headers as { get: (headerName: string) => unknown }).get ===
      "function"
  ) {
    const value = (headers as { get: (headerName: string) => unknown }).get(
      name,
    );
    return typeof value === "string" ? value : undefined;
  }

  const matchingHeader = Object.entries(headers).find(
    ([key]) => key.toLowerCase() === name,
  )?.[1];

  if (typeof matchingHeader === "number") {
    return String(matchingHeader);
  }

  return typeof matchingHeader === "string" ? matchingHeader : undefined;
}

export function isRateLimitError(error: unknown): boolean {
  const axiosError = error as AxiosError | undefined;
  return axiosError?.response?.status === 429 || axiosError?.status === 429;
}

function getRetryAfterDelayMs(error: unknown): number | undefined {
  const axiosError = error as AxiosError | undefined;
  const retryAfter = getHeaderValue(
    axiosError?.response?.headers,
    RETRY_AFTER_HEADER,
  );

  if (!retryAfter) {
    return undefined;
  }

  const seconds = Number(retryAfter);
  if (Number.isFinite(seconds) && seconds >= 0) {
    return Math.min(seconds * 1000, MAX_BASE_RETRY_DELAY_MS);
  }

  const retryAt = Date.parse(retryAfter);
  if (Number.isNaN(retryAt)) {
    return undefined;
  }

  return Math.min(Math.max(retryAt - Date.now(), 0), MAX_BASE_RETRY_DELAY_MS);
}

export function getRateLimitRetryDelayMs(
  failureCount: number,
  error: unknown,
): number {
  const exponentialDelay = Math.min(
    INITIAL_RETRY_DELAY_MS * 2 ** failureCount,
    MAX_BASE_RETRY_DELAY_MS,
  );
  const retryAfterDelay = getRetryAfterDelayMs(error);
  // Retry-After is a server-directed minimum; jitter only delays further.
  const minimumDelay = Math.max(retryAfterDelay ?? 0, exponentialDelay);
  const jitter = Math.floor(
    Math.random() * Math.min(exponentialDelay, MAX_JITTER_MS),
  );

  return minimumDelay + jitter;
}
