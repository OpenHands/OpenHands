import React from "react";
import {
  buildMarsBackendInput,
  closeMarsTunnel,
  fetchLatestMarsConversationId,
  getMarsBackendLocalPort,
  getMarsBridge,
  getMarsConnectionHost,
  openMarsTunnel,
  waitForMarsAgentServer,
} from "#/api/mars/mars-tunnel-backend";
import { prepareMarsSandbox } from "#/api/mars/mars-sandbox-setup";
import { resetBackendHealth } from "#/api/backend-registry/health-store";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import type { Backend } from "#/api/backend-registry/types";

export interface AttachMarsSessionParams {
  sessionId: string;
  /** Display name for the registered backend. */
  name: string;
  configId?: string;
  /** Fired once the session is reachable and the agent-server probe begins. */
  onTunnelReady?: () => void;
}

export interface AttachMarsSessionResult {
  backend: Backend;
  /** Where to land the user; null when the session has no conversation yet. */
  latestConversationId: string | null;
}

/**
 * Connect to the session and return the base URL to register: the loopback
 * tunnel listener (or, in the web build, the server's proxy path for it).
 */
async function openHealthyConnection(
  sessionId: string,
  localPort?: number,
): Promise<string> {
  const connection = await openMarsTunnel({ sessionId, localPort });
  const host = getMarsConnectionHost(connection);
  if (connection.status === "error" || host === undefined) {
    throw new Error(
      connection.error ?? `Failed to connect to session ${sessionId}`,
    );
  }
  return host;
}

/**
 * Registration/lifecycle wiring between MARS sessions and the backend
 * registry. `attach()` connects to the session, waits for its agent-server
 * to answer, and registers it as an ordinary local `Backend` (or re-points
 * the existing one for that session), making it active. `detach()` forgets
 * the connection and removes the Backend again.
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
      const host = await openHealthyConnection(sessionId);
      onTunnelReady?.();
      try {
        await waitForMarsAgentServer(host, { sessionId });
      } catch (error) {
        // A connection to a session nobody can use would keep failing the
        // health probe every tick (and, over the tunnel, hold a listener).
        await closeMarsTunnel(sessionId).catch(() => {});
        throw error;
      }
      await prepareMarsSandbox(host, sessionId);
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
          buildMarsBackendInput({ name, host, sessionId, configId }),
        ),
        latestConversationId,
      };
    },
    [addBackend, backends, setActive, updateBackend],
  );

  const detach = React.useCallback(
    async (backend: Backend): Promise<void> => {
      // Forget the connection before removing the Backend entry: if that
      // fails, the (now stale but still visible) Backend is a better failure
      // mode than silently leaking a tunnel listener behind a vanished one.
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
 * Connections live in the Electron main process and die with it, while the
 * Backend records pointing at them persist. Re-connect the *active* MARS
 * backend — at startup and whenever the user switches to one — and update
 * its host when the address changed. The previous local port is asked for
 * again so queries keyed to that host survive the restart, and a port
 * another process has since taken yields a new one.
 *
 * Only the active one: connecting resumes a paused session, so restoring
 * every registered session would wake (and bill) sandboxes the user is not
 * using.
 *
 * Returns true while the backend active at launch is still being
 * reconnected, so the bootstrap can wait for it instead of probing a dead
 * host and flashing the "unreachable backend" recovery screen.
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

    void openHealthyConnection(sessionId, getMarsBackendLocalPort(backend))
      .then(async (host) => {
        // A paused sandbox is woken by the connect; its agent-server
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
