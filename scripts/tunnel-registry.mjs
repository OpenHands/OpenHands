/**
 * Session-to-port tunnel registry (MARSOHS-1428).
 *
 * Tracks one MARS port-forward tunnel per attached session, since a user can
 * attach more than one MARS session as a backend at the same time (see
 * openhands-canvas-dataplane-design.md's "Session-to-port tracking"). Builds
 * directly on scripts/tunnel-client.mjs's startPortForwardTunnel() — this
 * module owns the session_id -> tunnel map, not the tunnel mechanics
 * themselves.
 *
 * Responsibilities (design doc scope for this ticket):
 *   - Port allocation with no collisions across concurrently attached
 *     sessions — free by construction: startPortForwardTunnel()'s default
 *     `localPort: 0` asks the OS for a free port, and the OS never hands out
 *     the same port to two independent listeners. No manual bookkeeping
 *     needed here beyond recording which port each session ended up with.
 *   - Reuse: calling attach() again for a session that already has a live
 *     tunnel returns that tunnel instead of opening a duplicate — after
 *     waking the session again, since it may have idle-paused (or been
 *     paused by hand) since the tunnel opened.
 *   - Targeted teardown: detach() stops one session's tunnel without
 *     touching any other session's.
 *   - Concurrent-attach dedup: two attach() calls for the same session
 *     firing before either has resolved share one in-flight attempt rather
 *     than racing to open two tunnels.
 *   - Waking a paused session before dialing it, via mars-session.mjs's
 *     ensureSessionAwake() — MARS auto-pauses idle sessions, and a tunnel
 *     that isn't open yet has generated no activity to have prevented that.
 *
 * Explicitly NOT this module's job:
 *   - Reconnect-after-app-restart detection. Because tunnels here run
 *     in-process (see tunnel-client.mjs's header comment for why), an app
 *     restart unconditionally kills every tunnel — there is no "is the old
 *     one still alive?" check to perform, unlike a design where tunnels ran
 *     as independent subprocesses that could outlive a restart. This
 *     registry always starts empty on a fresh process. What DOES persist
 *     across restarts is the `Backend` record in Canvas's UI-facing
 *     backend-registry (a separate system) — comparing "what's persisted"
 *     against "what's actually attached right now" and deciding whether to
 *     re-attach (optionally passing the previous `localPort` back into
 *     attach() to reuse it) or surface as disconnected is Ticket 3's job
 *     (MARSOHS-1429), which owns that comparison. This registry only
 *     provides the primitive that makes it possible: attach() accepts an
 *     optional `localPort` to re-request a specific port, falling back to
 *     an OS-picked one when something else has taken it in the meantime.
 *   - Wiring into Electron's app lifecycle (app quit, session pause/end).
 *     That's also Ticket 3 — this module is a standalone, directly testable
 *     piece, following the same shape as Ticket 1.
 */

import { startPortForwardTunnel } from "./tunnel-client.mjs";
import { ensureSessionAwake } from "./mars-session.mjs";
import { createMarsApiClient } from "./mars-api.mjs";

/**
 * @param {{ sessionId: string, apiUrl?: string, getAccessToken: () => string | null }} options
 */
function wakeSession({ sessionId, apiUrl, getAccessToken }) {
  return ensureSessionAwake({
    api: createMarsApiClient({
      baseUrl: apiUrl,
      getToken: async () => getAccessToken(),
    }),
    sessionId,
  });
}

/**
 * @typedef {{
 *   sessionId: string,
 *   status: "connecting" | "connected" | "error",
 *   remotePort: number,
 *   localPort: number | undefined,
 *   error: string | undefined,
 *   upstreamFailure: { closeCode: number | null, httpStatus: number | null, message: string } | null,
 * }} TunnelEntrySnapshot
 */

/**
 * @param {object} [options]
 * @param {typeof startPortForwardTunnel} [options.startTunnel] Override for tests
 * @param {(options: { sessionId: string, apiUrl?: string, getAccessToken: () => string | null }) => Promise<unknown>} [options.ensureAwake]
 *   Resolves once the session is READY. Defaults to ensureSessionAwake over
 *   a client authenticated with the tunnel's own token.
 */
export function createTunnelRegistry({
  startTunnel = startPortForwardTunnel,
  ensureAwake = wakeSession,
} = {}) {
  /** @type {Map<string, { status: "connecting" | "connected" | "error", remotePort: number, apiUrl: string | undefined, getAccessToken: () => string | null, owner: string | undefined, tunnel: object | null, error: Error | null, promise: Promise<void> | null }>} */
  const entries = new Map();

  function snapshot(sessionId) {
    const entry = entries.get(sessionId);
    if (!entry) return undefined;
    return {
      sessionId,
      status: entry.status,
      remotePort: entry.remotePort,
      localPort: entry.tunnel?.localPort,
      error: entry.error?.message,
      upstreamFailure: entry.tunnel?.getLastUpstreamFailure?.() ?? null,
    };
  }

  /**
   * A restored Backend asks for the port it had before the restart, which
   * another process may have bound since.
   */
  async function startOnPreferredPort(options) {
    try {
      return await startTunnel(options);
    } catch (error) {
      if (!options.localPort || error?.code !== "EADDRINUSE") throw error;
      return startTunnel({ ...options, localPort: 0 });
    }
  }

  /**
   * Attach (or reuse) the tunnel for one session. Resolves once the tunnel
   * is up, or rejects if dialing it failed — either way, the outcome is
   * also queryable afterward via get(sessionId).
   *
   * @param {object} options
   * @param {string} options.sessionId
   * @param {number} options.remotePort
   * @param {() => string | null} options.getAccessToken Read on every dial,
   *   so a tunnel keeps using whichever credential it was opened with and
   *   stops working once that credential is gone.
   * @param {string} [options.apiUrl]
   * @param {number} [options.localPort] Reuse a specific local port (e.g. the
   *   one a previously persisted Backend recorded) instead of letting the OS
   *   pick a new one.
   * @param {string} [options.owner] Opaque tag for detachOwnedBy(), e.g. the
   *   credential the tunnel dials with.
   * @returns {Promise<TunnelEntrySnapshot>}
   */
  async function attach(options) {
    const { sessionId, remotePort, getAccessToken, apiUrl, localPort, owner } =
      options;
    if (!sessionId) {
      throw new Error("sessionId is required");
    }

    const existing = entries.get(sessionId);
    if (existing) {
      if (existing.status === "connected") {
        await ensureAwake({
          sessionId,
          apiUrl: existing.apiUrl,
          getAccessToken: existing.getAccessToken,
        });
        return snapshot(sessionId);
      }
      if (existing.status === "connecting") {
        // Piggyback on the in-flight attempt rather than racing it — once it
        // settles (either way), re-run attach() to either reuse the tunnel
        // it produced or retry after its failure.
        await existing.promise.catch(() => {});
        return attach(options);
      }
      // status === "error": fall through and retry below.
    }

    const entry = {
      status: "connecting",
      remotePort,
      apiUrl,
      getAccessToken,
      owner,
      tunnel: null,
      error: null,
      promise: null,
    };
    entries.set(sessionId, entry);

    entry.promise = (async () => {
      // MARS auto-pauses idle sessions, and a tunnel that isn't open yet
      // can't have generated any activity to prevent that — so a session
      // being attached to for the first time in a while may need waking
      // before dialing it can succeed at all.
      await ensureAwake({ sessionId, apiUrl, getAccessToken });
      const tunnel = await startOnPreferredPort({
        sessionId,
        remotePort,
        getAccessToken,
        apiUrl,
        localPort,
      });
      entry.tunnel = tunnel;
      entry.status = "connected";
    })().catch((err) => {
      entry.error = err;
      entry.status = "error";
      throw err;
    });

    await entry.promise;
    return snapshot(sessionId);
  }

  /** @returns {TunnelEntrySnapshot | undefined} */
  function get(sessionId) {
    return snapshot(sessionId);
  }

  /** @returns {TunnelEntrySnapshot[]} */
  function list() {
    return [...entries.keys()].map(snapshot);
  }

  /** Tear down one session's tunnel, if any, without affecting others. */
  async function detach(sessionId) {
    const entry = entries.get(sessionId);
    if (!entry) return;
    entries.delete(sessionId);
    // Let an in-flight attach() settle first so its tunnel (if it succeeds)
    // doesn't get created after we've already "detached" and get orphaned.
    if (entry.status === "connecting") {
      await entry.promise.catch(() => {});
    }
    entry.tunnel?.stop();
  }

  async function detachAll() {
    await Promise.all(
      [...entries.keys()].map((sessionId) => detach(sessionId)),
    );
  }

  /** Tear down every tunnel attached with the given `owner`. */
  async function detachOwnedBy(owner) {
    await Promise.all(
      [...entries]
        .filter(([, entry]) => entry.owner === owner)
        .map(([sessionId]) => detach(sessionId)),
    );
  }

  return { attach, get, list, detach, detachAll, detachOwnedBy };
}
