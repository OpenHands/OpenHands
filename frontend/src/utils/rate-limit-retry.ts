import { AxiosError } from "axios";

const DEFAULT_RETRY_AFTER_MS = 1000;
const MAX_RETRY_DELAY_MS = 60_000;
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

export function getRateLimitRetryDelayMs(error: unknown): number {
  const axiosError = error as AxiosError | undefined;
  const retryAfter = getHeaderValue(
    axiosError?.response?.headers,
    RETRY_AFTER_HEADER,
  );

  if (!retryAfter) {
    return DEFAULT_RETRY_AFTER_MS;
  }

  const seconds = Number(retryAfter);
  if (Number.isFinite(seconds) && seconds >= 0) {
    return Math.min(seconds * 1000, MAX_RETRY_DELAY_MS);
  }

  const retryAt = Date.parse(retryAfter);
  if (Number.isNaN(retryAt)) {
    return DEFAULT_RETRY_AFTER_MS;
  }

  return Math.min(Math.max(retryAt - Date.now(), 0), MAX_RETRY_DELAY_MS);
}
