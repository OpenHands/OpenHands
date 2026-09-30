import React from "react";
import {
  buildMarsBackendHost,
  buildMarsBackendInput,
  closeMarsTunnel,
  fetchLatestMarsConversationId,
  getMarsBackendLocalPort,
  getMarsBridge,
  openMarsTunnel,
  waitForMarsAgentServer,
} from "#/api/mars/mars-tunnel-backend";
import { resetBackendHealth } from "#/api/backend-registry/health-store";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import type { Backend } from "#/api/backend-registry/types";

export interface AttachMarsSessionParams {
  sessionId: string;
  /** Display name for the registered backend. */
  name: string;
  configId?: string;
  /** Fired once the tunnel is up and the agent-server probe begins. */
  onTunnelReady?: () => void;
}

export interface AttachMarsSessionResult {
  backend: Backend;
  /** Where to land the user; null when the session has no conversation yet. */
  latestConversationId: string | null;
}

async function openHealthyTunnel(
  sessionId: string,
  localPort?: number,
): Promise<number> {
  const tunnel = await openMarsTunnel({ sessionId, localPort });
  if (tunnel.status === "error" || tunnel.localPort === undefined) {
    throw new Error(
      tunnel.error ?? `Failed to open a tunnel for session ${sessionId}`,
    );
  }
  return tunnel.localPort;
}

/**
 * Registration/lifecycle wiring between MARS sessions and the backend
 * registry. `attach()` opens the session's tunnel, waits for its
 * agent-server to answer, and registers it as an ordinary local `Backend`
 * (or re-points the existing one for that session), making it active.
 * `detach()` tears the tunnel down and removes the Backend again.
 */
export function useMarsTunnelBackend() {
  const { backends, addBackend, removeBackend, updateBackend, setActive } =
    useActiveBackendContext();

  const attach = React.useCallback(
    async ({
      sessionId,
      name,
      configId,
      onTunnelReady,
    }: AttachMarsSessionParams): Promise<AttachMarsSessionResult> => {
      const localPort = await openHealthyTunnel(sessionId);
      const host = buildMarsBackendHost(localPort);
      onTunnelReady?.();
      try {
        await waitForMarsAgentServer(host, { sessionId });
      } catch (error) {
        // A listener bound to a session nobody can use would keep failing
        // the health probe every tick.
        await closeMarsTunnel(sessionId).catch(() => {});
        throw error;
      }
      // Read before the registry switch below, while nothing else is
      // pointed at this host yet.
      const latestConversationId = await fetchLatestMarsConversationId(host);

      const existing = backends.find((b) => b.marsSessionId === sessionId);
      if (existing) {
        updateBackend(existing.id, { host, name });
        setActive(existing.id);
        return {
          backend: { ...existing, host, name },
          latestConversationId,
        };
      }
      return {
        backend: addBackend(
          buildMarsBackendInput({ name, localPort, sessionId, configId }),
        ),
        latestConversationId,
      };
    },
    [addBackend, backends, setActive, updateBackend],
  );

  const detach = React.useCallback(
    async (backend: Backend): Promise<void> => {
      // Close the tunnel before removing the Backend entry: if closing fails,
      // the (now stale but still visible) Backend is a better failure mode
      // than silently leaking an open local listener behind a vanished one.
      if (backend.marsSessionId) {
        await closeMarsTunnel(backend.marsSessionId);
      }
      removeBackend(backend.id);
    },
    [removeBackend],
  );

  return { attach, detach };
}

/**
 * Tunnels live in the Electron main process and die with it, while the
 * Backend records pointing at them persist. Re-open the *active* MARS
 * backend's tunnel — at startup and whenever the user switches to one —
 * on its previous local port when still free, so every query keyed to that
 * host survives the restart.
 *
 * Only the active one: opening a tunnel resumes a paused session, so
 * restoring every registered session would wake (and bill) sandboxes the
 * user is not using.
 *
 * Returns true while the tunnel of the backend active at launch is still
 * being restored, so the bootstrap can wait for it instead of probing a dead
 * port and flashing the "unreachable backend" recovery screen.
 */
export function useRestoreMarsTunnels(): boolean {
  const { active, updateBackend } = useActiveBackendContext();
  const { backend } = active;
  const attempted = React.useRef(new Set<string>());
  const [launchRestoreId, setLaunchRestoreId] = React.useState(() =>
    getMarsBridge() && backend.marsSessionId ? backend.id : null,
  );

  React.useEffect(() => {
    const sessionId = backend.marsSessionId;
    if (!getMarsBridge() || !sessionId || attempted.current.has(backend.id)) {
      return;
    }
    attempted.current.add(backend.id);

    void openHealthyTunnel(sessionId, getMarsBackendLocalPort(backend))
      .then(async (localPort) => {
        const host = buildMarsBackendHost(localPort);
        // A paused sandbox wakes on the tunnel dial; its agent-server
        // answers only once booted.
        await waitForMarsAgentServer(host, { sessionId });
        if (host === backend.host) {
          resetBackendHealth(backend.id);
        } else {
          updateBackend(backend.id, { host });
        }
      })
      .catch(() => {
        // Signed out, session ended, or MARS unreachable: the health dot
        // already reports the backend as offline, and the Managed Agents
        // screen offers reconnecting. Allow a retry on the next switch.
        attempted.current.delete(backend.id);
      })
      .finally(() => {
        setLaunchRestoreId((id) => (id === backend.id ? null : id));
      });
  }, [backend, updateBackend]);

  return launchRestoreId !== null && launchRestoreId === backend.id;
}
