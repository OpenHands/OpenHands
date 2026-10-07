/**
 * MARS public ingress: resolve the HTTPS URL of a session's OpenHands Agent
 * Server.
 *
 * harness-api publishes one URL per session (`GET
 * /v2/agents/sessions/{id}/ingress`) that fronts guest port 8000 through the
 * microVM's public ingress. The renderer talks to it directly — REST and
 * WebSocket — so no local listener is needed, unlike the port-forward tunnel
 * in tunnel-client.mjs. The URL is PAT-authenticated on every request; the
 * main process injects that header (see mars-tunnel-bridge.mjs
 * `registerRequestAuth`), so the renderer never holds the token.
 *
 * Lifecycle facts this module is built around (harness-api contract):
 *   - The endpoint answers 409 for a PAUSED session, so the session is woken
 *     first (ensureSessionAwake), exactly as the tunnel path does.
 *   - A freshly published URL starts PENDING and becomes READY once the
 *     gateway route is up; there is no failed state, so this polls.
 *   - 501 means this session can never have a URL (pre-ingress sandbox, a
 *     microVM launched before the port was published, non-OpenHands agent).
 *     That is the one error callers fall back to the tunnel on, so it is
 *     surfaced as its own type.
 *   - The hostname is revoked on pause and lock, and changes after rollback,
 *     so a URL must be re-resolved on every connect rather than cached.
 */

import { setTimeout as sleep } from "node:timers/promises";
import { INGRESS_URL_STATE_READY, MarsApiError } from "./mars-api.mjs";
import { ensureSessionAwake } from "./mars-session.mjs";

const DEFAULT_POLL_INTERVAL_MS = 2_000;
const DEFAULT_TIMEOUT_MS = 120_000;
const HTTP_NOT_IMPLEMENTED = 501;

/** The session cannot have a public URL; the caller should use the tunnel. */
export class MarsIngressUnsupportedError extends Error {
  constructor(message, { cause = null } = {}) {
    super(message);
    this.name = "MarsIngressUnsupportedError";
    this.status = HTTP_NOT_IMPLEMENTED;
    if (cause) this.cause = cause;
  }
}

/**
 * @typedef {{
 *   url: string,
 *   ingressUrlId: string,
 *   port: number,
 *   state: string,
 * }} ResolvedIngress
 */

async function waitUntilPublished(api, sessionId, pollIntervalMs, signal) {
  for (;;) {
    let ingress;
    try {
      ingress = await api.getIngressURL(sessionId, { signal });
    } catch (error) {
      if (
        error instanceof MarsApiError &&
        error.status === HTTP_NOT_IMPLEMENTED
      ) {
        throw new MarsIngressUnsupportedError(error.message, { cause: error });
      }
      throw error;
    }
    if (!ingress?.url) {
      throw new Error(`DigitalOcean returned no ingress URL for ${sessionId}.`);
    }
    if (ingress.state === INGRESS_URL_STATE_READY) {
      return {
        url: ingress.url,
        ingressUrlId: ingress.ingress_url_id,
        port: ingress.port,
        state: ingress.state,
      };
    }
    await sleep(pollIntervalMs, undefined, { signal });
  }
}

/**
 * Resolve the READY public URL for a session, waking it first if paused.
 * Rejects with {@link MarsIngressUnsupportedError} when the session can never
 * have one, with the harness-api error for anything else, and with a timeout
 * error if the URL does not become READY within `timeoutMs`.
 *
 * @param {object} options
 * @param {Pick<ReturnType<typeof import("./mars-api.mjs").createMarsApiClient>, "getSession" | "resumeSession" | "getIngressURL">} options.api
 * @param {string} options.sessionId
 * @param {typeof ensureSessionAwake} [options.ensureAwake] Override for tests
 * @param {number} [options.pollIntervalMs]
 * @param {number} [options.timeoutMs] Bounds the wake and the publish wait together.
 * @returns {Promise<ResolvedIngress>}
 */
export async function resolveIngressURL({
  api,
  sessionId,
  ensureAwake = ensureSessionAwake,
  pollIntervalMs = DEFAULT_POLL_INTERVAL_MS,
  timeoutMs = DEFAULT_TIMEOUT_MS,
}) {
  const deadline = AbortSignal.timeout(timeoutMs);
  try {
    await ensureAwake({ api, sessionId, pollIntervalMs, timeoutMs });
    return await waitUntilPublished(api, sessionId, pollIntervalMs, deadline);
  } catch (error) {
    if (deadline.aborted && !(error instanceof MarsIngressUnsupportedError)) {
      throw new Error(
        `Timed out waiting for the public URL of session ${sessionId}.`,
        { cause: error },
      );
    }
    throw error;
  }
}
