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
 * Built on mars-api.mjs's client so these calls carry the same auth, device
 * id, and per-request timeout as every other MARS request.
 */

import { setTimeout as sleep } from "node:timers/promises";
import {
  SESSION_STATUS_PAUSED,
  SESSION_STATUS_READY,
  isTerminalSessionStatus,
} from "./mars-api.mjs";

const DEFAULT_POLL_INTERVAL_MS = 2_000;
const DEFAULT_TIMEOUT_MS = 120_000;

async function readSession(api, sessionId, signal) {
  const session = await api.getSession(sessionId, { signal });
  if (!session) {
    throw new Error(`DigitalOcean returned no session ${sessionId}.`);
  }
  return session;
}

async function waitUntilReady(api, sessionId, pollIntervalMs, signal) {
  let session = await readSession(api, sessionId, signal);
  if (isTerminalSessionStatus(session.status)) {
    throw new Error(
      `Session ${sessionId} is ${session.status} and cannot be connected to.`,
    );
  }
  if (session.status === SESSION_STATUS_READY) {
    return session;
  }
  if (session.status === SESSION_STATUS_PAUSED) {
    await api.resumeSession(sessionId, { signal });
  }

  for (;;) {
    await sleep(pollIntervalMs, undefined, { signal });
    session = await readSession(api, sessionId, signal);
    if (session.status === SESSION_STATUS_READY) {
      return session;
    }
    if (isTerminalSessionStatus(session.status)) {
      throw new Error(
        `Session ${sessionId} became ${session.status} while resuming.`,
      );
    }
  }
}

/**
 * Resolve once the session is SESSION_STATUS_READY, resuming it first if
 * it's paused. Throws if the session is (or becomes) terminal, or if it
 * doesn't reach ready within timeoutMs. The deadline bounds the whole wait,
 * including any request still in flight when it passes.
 *
 * @param {object} options
 * @param {Pick<ReturnType<typeof import("./mars-api.mjs").createMarsApiClient>, "getSession" | "resumeSession">} options.api
 * @param {string} options.sessionId
 * @param {number} [options.pollIntervalMs]
 * @param {number} [options.timeoutMs]
 * @returns {Promise<{status: string, [key: string]: any}>} the session, once ready
 */
export async function ensureSessionAwake({
  api,
  sessionId,
  pollIntervalMs = DEFAULT_POLL_INTERVAL_MS,
  timeoutMs = DEFAULT_TIMEOUT_MS,
}) {
  const deadline = AbortSignal.timeout(timeoutMs);
  try {
    return await waitUntilReady(api, sessionId, pollIntervalMs, deadline);
  } catch (error) {
    if (deadline.aborted) {
      throw new Error(
        `Timed out waiting for session ${sessionId} to become ready.`,
        { cause: error },
      );
    }
    throw error;
  }
}
