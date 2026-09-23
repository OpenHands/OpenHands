import React from "react";
import {
  buildMarsBackendInput,
  closeMarsTunnel,
  openMarsTunnel,
  type OpenMarsTunnelParams,
} from "#/api/mars/mars-tunnel-backend";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import type { Backend } from "#/api/backend-registry/types";

export interface AttachMarsSessionParams extends OpenMarsTunnelParams {
  /** Display name for the registered backend, e.g. the session's name. */
  name: string;
  /** Passed through to the guest agent-server, if it requires one. */
  apiKey?: string;
}

/**
 * The registration/lifecycle wiring behind whatever "Add Backend" DO tile
 * eventually lands (MARSOHS-1195, tracked and built separately) — not a UI
 * itself. Once a session's tunnel is healthy, `attach()` registers it as an
 * ordinary local `Backend` through the existing backend-registry code path
 * (`useActiveBackendContext().addBackend`, unmodified); `detach()` tears the
 * tunnel down and removes that Backend again, for session end, pause, or
 * whatever else the caller decides warrants it — reconnect-state surfacing
 * itself needs no extra code here, since a registered local Backend already
 * gets the standard health-polling treatment (`useBackendsHealth`) any other
 * local backend gets, and its probes simply start failing once the tunnel's
 * local port stops answering.
 */
export function useMarsTunnelBackend() {
  const { addBackend, removeBackend } = useActiveBackendContext();

  const attach = React.useCallback(
    async ({
      name,
      apiKey,
      ...tunnelParams
    }: AttachMarsSessionParams): Promise<Backend> => {
      const tunnel = await openMarsTunnel(tunnelParams);
      if (tunnel.status === "error" || tunnel.localPort === undefined) {
        throw new Error(
          tunnel.error ??
            `Failed to open a tunnel for session ${tunnelParams.sessionId}`,
        );
      }
      return addBackend(
        buildMarsBackendInput({
          name,
          localPort: tunnel.localPort,
          apiKey,
        }),
      );
    },
    [addBackend],
  );

  const detach = React.useCallback(
    async (backendId: string, sessionId: string): Promise<void> => {
      // Close the tunnel before removing the Backend entry: if closing fails,
      // the (now stale but still visible) Backend is a better failure mode
      // than silently leaking an open local listener behind a vanished one.
      await closeMarsTunnel(sessionId);
      removeBackend(backendId);
    },
    [removeBackend],
  );

  return { attach, detach };
}
