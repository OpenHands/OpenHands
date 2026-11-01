import React from "react";
import {
  buildMarsBackendInput,
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
import {
  getUsableConnectionId,
  useMarsAuthState,
} from "#/hooks/query/use-mars-agents";
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
 * Connect and wait for the session's agent-server to answer. With
 * `closeOnFailure`, a connection whose agent-server never answers is closed:
 * registering a session nobody can use would keep failing the health probe
 * every tick (and, over the tunnel, hold a listener). Restoring leaves it
 * open — a slow wake can still recover through the health poll, the
 * unreachable screen can report the agent-server's own error, and the
 * tunnel may be shared with a concurrent attach.
 */
async function connectAndWait(
  sessionId: string,
  {
    localPort,
    onTunnelReady,
    closeOnFailure = false,
  }: {
    localPort?: number;
    onTunnelReady?: () => void;
    closeOnFailure?: boolean;
  } = {},
): Promise<string> {
  const host = await openHealthyConnection(sessionId, localPort);
  onTunnelReady?.();
  try {
    await waitForMarsAgentServer(host, { sessionId });
  } catch (error) {
    if (closeOnFailure) {
      await getMarsBridge()
        ?.closeTunnel(sessionId)
        .catch(() => {});
    }
    throw error;
  }
  return host;
}

/**
 * Backend ids `attach()` has just connected, so the restore effect that sees
 * them become active does not connect (and wait for) them all over again.
 */
const freshlyAttached = new Set<string>();

/**
 * Registration/lifecycle wiring between MARS sessions and the backend
 * registry. `attach()` connects to the session, waits for its agent-server
 * to answer, and registers it as an ordinary local `Backend` (or re-points
 * the existing one for that session), making it active. `detach()` removes
 * the Backend again, which forgets the connection.
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
      const host = await connectAndWait(sessionId, {
        onTunnelReady,
        closeOnFailure: true,
      });
      await prepareMarsSandbox(host, sessionId);
      // Read before the registry switch below, while nothing else is
      // pointed at this host yet.
      const latestConversationId = await fetchLatestMarsConversationId(host);

      const existing = backends.find((b) => b.marsSessionId === sessionId);
      if (existing) {
        freshlyAttached.add(existing.id);
        updateBackend(existing.id, { host, name });
        // The restore effect is skipped for it, and `updateBackend` resets
        // health only when the host changes (it never does in the web build).
        resetBackendHealth(existing.id);
        setActive(existing.id);
        return {
          backend: { ...existing, host, name },
          latestConversationId,
        };
      }
      const backend = addBackend(
        buildMarsBackendInput({ name, host, sessionId, configId }),
      );
      freshlyAttached.add(backend.id);
      return { backend, latestConversationId };
    },
    [addBackend, backends, setActive, updateBackend],
  );

  // Removing the Backend also closes its session's tunnel (see
  // `removeBackend`), so every removal path forgets the connection.
  const detach = React.useCallback(
    async (backend: Backend): Promise<void> => {
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
  // Connecting needs a signed-in DigitalOcean connection. The web build keeps
  // it only in server memory, so after a restart the user signs in again; the
  // connection appearing re-runs the effect below and reconnects then.
  const authQuery = useMarsAuthState();
  const isAuthPending = authQuery.isPending;
  const connectionId = getUsableConnectionId(authQuery.data);

  React.useEffect(() => {
    const sessionId = backend.marsSessionId;
    if (!getMarsBridge() || !sessionId) return;
    // attach() already connected it and waited for its agent-server.
    if (freshlyAttached.delete(backend.id)) {
      attempted.current.add(backend.id);
      return;
    }
    if (isAuthPending) return;
    if (!connectionId) {
      // Nothing to connect with yet; don't hold the bootstrap for it.
      setLaunchRestoreId((id) => (id === backend.id ? null : id));
      return;
    }
    if (attempted.current.has(backend.id)) return;
    attempted.current.add(backend.id);

    // A paused sandbox is woken by the connect; its agent-server answers
    // only once booted.
    void connectAndWait(sessionId, {
      localPort: getMarsBackendLocalPort(backend),
    })
      .then((host) => {
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
  }, [backend, updateBackend, isAuthPending, connectionId]);

  return launchRestoreId !== null && launchRestoreId === backend.id;
}
