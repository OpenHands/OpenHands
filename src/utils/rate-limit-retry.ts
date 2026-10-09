import { AxiosError } from "axios";

const BASE_DELAY_MS = 1000;
const MAX_DELAY_MS = 30000;
/**
 * Extra backoff multiplier applied on top of the normal exponential curve
 * once a request has actually been rate-limited (429). A plain network
 * blip should back off quickly; a server telling us to slow down should
 * back off harder so a burst of tabs/queries don't all retry in lockstep
 * and immediately re-trip the limiter.
 */
const RATE_LIMIT_BACKOFF_MULTIPLIER = 2;

/**
 * True for a 429 from either transport this app talks to a backend with:
 * Axios (agent-server / legacy cloud calls) or the duck-typed `{ status }`
 * shape `@openhands/typescript-client`'s `HttpError` uses for the cloud
 * proxy. `HttpError` isn't imported here to keep this usable from any
 * query regardless of which client raised the error.
 */
export function isRateLimitError(error: unknown): boolean {
  if (error instanceof AxiosError) {
    return error.response?.status === 429 || error.status === 429;
  }
  if (error && typeof error === "object" && "status" in error) {
    return (error as { status?: unknown }).status === 429;
  }
  return false;
}

/**
 * Default `retryDelay` for the shared query client: exponential backoff
 * capped at 30s, with full jitter so concurrent queries (e.g. every open
 * browser tab refetching on window focus at once) don't retry in sync and
 * re-create the burst that got them rate-limited in the first place.
 * Doubles the base delay on a confirmed 429 before applying jitter.
 */
export function getQueryRetryDelay(
  attemptIndex: number,
  error: unknown,
): number {
  const multiplier = isRateLimitError(error)
    ? RATE_LIMIT_BACKOFF_MULTIPLIER
    : 1;
  const capped = Math.min(
    BASE_DELAY_MS * multiplier * 2 ** attemptIndex,
    MAX_DELAY_MS,
  );
  return Math.round(capped * (0.5 + Math.random() * 0.5));
}
