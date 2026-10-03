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
 *     tunnel returns that tunnel instead of opening a duplicate.
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
 *     optional `localPort` to re-request a specific port instead of always
 *     letting the OS pick a new one.
 *   - Wiring into Electron's app lifecycle (app quit, session pause/end).
 *     That's also Ticket 3 — this module is a standalone, directly testable
 *     piece, following the same shape as Ticket 1.
 */

import { startPortForwardTunnel } from "./tunnel-client.mjs";
import { ensureSessionAwake } from "./mars-session.mjs";

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
 * @param {typeof ensureSessionAwake} [options.ensureAwake] Override for tests
 */
export function createTunnelRegistry({
  startTunnel = startPortForwardTunnel,
  ensureAwake = ensureSessionAwake,
} = {}) {
  /** @type {Map<string, { status: "connecting" | "connected" | "error", remotePort: number, tunnel: object | null, error: Error | null, promise: Promise<void> | null }>} */
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
   * Attach (or reuse) the tunnel for one session. Resolves once the tunnel
   * is up, or rejects if dialing it failed — either way, the outcome is
   * also queryable afterward via get(sessionId).
   *
   * @param {object} options
   * @param {string} options.sessionId
   * @param {number} options.remotePort
   * @param {string} options.accessToken
   * @param {string} [options.apiUrl]
   * @param {number} [options.localPort] Reuse a specific local port (e.g. the
   *   one a previously persisted Backend recorded) instead of letting the OS
   *   pick a new one.
   * @returns {Promise<TunnelEntrySnapshot>}
   */
  async function attach({
    sessionId,
    remotePort,
    accessToken,
    apiUrl,
    localPort,
  }) {
    if (!sessionId) {
      throw new Error("sessionId is required");
    }

    const existing = entries.get(sessionId);
    if (existing) {
      if (existing.status === "connected") {
        return snapshot(sessionId);
      }
      if (existing.status === "connecting") {
        // Piggyback on the in-flight attempt rather than racing it — once it
        // settles (either way), re-run attach() to either reuse the tunnel
        // it produced or retry after its failure.
        await existing.promise.catch(() => {});
        return attach({
          sessionId,
          remotePort,
          accessToken,
          apiUrl,
          localPort,
        });
      }
      // status === "error": fall through and retry below.
    }

    const entry = {
      status: "connecting",
      remotePort,
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
      await ensureAwake({ apiUrl, sessionId, accessToken });
      const tunnel = await startTunnel({
        sessionId,
        remotePort,
        accessToken,
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

  return { attach, get, list, detach, detachAll };
}
