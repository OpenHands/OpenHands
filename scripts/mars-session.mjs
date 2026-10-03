/**
 * MARS session lifecycle helper: resume a paused session before its tunnel
 * is dialed.
 *
 * MARS auto-pauses idle sessions (HARNESS_IDLE_PAUSE_TIMEOUT), and — per
 * openhands-canvas-dataplane-design.md's "Inherited risk" section — an open
 * but quiet port-forward tunnel does not currently count as activity, so a
 * long-lived chat session WILL hit this, not just as an edge case. Mirrors
 * doctl's ensureSessionAwakeForPortForward (agent_port_forward.go), which
 * resumes a paused session before opening a tunnel the same way `launch`/
 * `attach` already do.
 *
 * Endpoints (godo's hosted_agents.go, the doctl client's own backing library):
 *   GET  /v2/agents/sessions/{id}          -> { session: { status, ... } }
 *   POST /v2/agents/sessions/{id}/resume   -> 204, no body
 */

const READY = "SESSION_STATUS_READY";
const PAUSED = "SESSION_STATUS_PAUSED";

// Statuses a session can never leave — waiting them out is pointless.
const TERMINAL_STATUSES = new Set([
  "SESSION_STATUS_DESTROYING",
  "SESSION_STATUS_DESTROYED",
  "SESSION_STATUS_FAILED",
]);

const DEFAULT_API_URL = "https://api.digitalocean.com/";
const DEFAULT_POLL_INTERVAL_MS = 2_000;
const DEFAULT_TIMEOUT_MS = 120_000;

function sessionUrl(apiUrl, sessionId, suffix = "") {
  const url = new URL(apiUrl);
  url.pathname =
    url.pathname.replace(/\/$/, "") +
    `/v2/agents/sessions/${sessionId}${suffix}`;
  return url.toString();
}

async function getSession(apiUrl, sessionId, accessToken, fetchImpl) {
  const res = await fetchImpl(sessionUrl(apiUrl, sessionId), {
    headers: { Authorization: `Bearer ${accessToken}` },
  });
  if (!res.ok) {
    throw new Error(`Failed to get session ${sessionId}: HTTP ${res.status}`);
  }
  const body = await res.json();
  return body.session;
}

async function resumeSession(apiUrl, sessionId, accessToken, fetchImpl) {
  const res = await fetchImpl(sessionUrl(apiUrl, sessionId, "/resume"), {
    method: "POST",
    headers: { Authorization: `Bearer ${accessToken}` },
  });
  if (!res.ok) {
    throw new Error(
      `Failed to resume session ${sessionId}: HTTP ${res.status}`,
    );
  }
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Resolve once the session is SESSION_STATUS_READY, resuming it first if
 * it's paused. Throws if the session is (or becomes) terminal, or if it
 * doesn't reach ready within timeoutMs.
 *
 * @param {object} options
 * @param {string} [options.apiUrl]
 * @param {string} options.sessionId
 * @param {string} options.accessToken
 * @param {number} [options.pollIntervalMs]
 * @param {number} [options.timeoutMs]
 * @param {typeof fetch} [options.fetchImpl] Override for tests
 * @returns {Promise<{status: string, [key: string]: any}>} the session, once ready
 */
export async function ensureSessionAwake({
  apiUrl = DEFAULT_API_URL,
  sessionId,
  accessToken,
  pollIntervalMs = DEFAULT_POLL_INTERVAL_MS,
  timeoutMs = DEFAULT_TIMEOUT_MS,
  fetchImpl = fetch,
}) {
  let session = await getSession(apiUrl, sessionId, accessToken, fetchImpl);
  if (TERMINAL_STATUSES.has(session.status)) {
    throw new Error(
      `Session ${sessionId} is ${session.status} and cannot be connected to.`,
    );
  }
  if (session.status === READY) {
    return session;
  }
  if (session.status === PAUSED) {
    await resumeSession(apiUrl, sessionId, accessToken, fetchImpl);
  }

  const deadline = Date.now() + timeoutMs;
  for (;;) {
    await delay(pollIntervalMs);
    session = await getSession(apiUrl, sessionId, accessToken, fetchImpl);
    if (session.status === READY) {
      return session;
    }
    if (TERMINAL_STATUSES.has(session.status)) {
      throw new Error(
        `Session ${sessionId} became ${session.status} while resuming.`,
      );
    }
    if (Date.now() >= deadline) {
      throw new Error(
        `Timed out waiting for session ${sessionId} to become ready.`,
      );
    }
  }
}
