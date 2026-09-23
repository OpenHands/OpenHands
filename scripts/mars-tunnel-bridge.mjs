/**
 * Wires scripts/tunnel-registry.mjs into Electron's IPC layer (MARSOHS-1429).
 *
 * Channel names mirror the shape of a teammate's independent, further-along
 * MARS integration (window.marsBridge / mars:openTunnel / mars:closeTunnel)
 * so that whichever "Add Backend" UI eventually lands (MARSOHS-1195, tracked
 * separately) has less to adapt to reach this wiring. Only the tunnel-open/
 * close surface is adopted here — that fork's auth/OAuth/credential-storage
 * and second ("OAH") data plane are a separate, much larger scope this
 * ticket does not include.
 *
 * v1 scope difference from that fork: it has no credential store of its own
 * yet, so `openTunnel` takes the bearer token as an explicit argument from
 * its caller rather than sourcing one internally. Whatever lands the actual
 * OAuth/PAT UI can narrow this call to take just a sessionId later, the same
 * way that fork's does.
 */

import { createTunnelRegistry } from "./tunnel-registry.mjs";

export const MARS_TUNNEL_IPC = {
  openTunnel: "mars:openTunnel",
  closeTunnel: "mars:closeTunnel",
  getTunnel: "mars:getTunnel",
};

/**
 * @param {object} [options]
 * @param {ReturnType<typeof createTunnelRegistry>} [options.registry] Override for tests
 */
export function createMarsTunnelBridge({ registry = createTunnelRegistry() } = {}) {
  return {
    registry,

    /** @param {import("electron").IpcMain} ipcMain */
    registerIpc(ipcMain) {
      ipcMain.handle(MARS_TUNNEL_IPC.openTunnel, (_event, params) =>
        registry.attach(params),
      );
      ipcMain.handle(MARS_TUNNEL_IPC.closeTunnel, (_event, sessionId) =>
        registry.detach(sessionId),
      );
      ipcMain.handle(MARS_TUNNEL_IPC.getTunnel, (_event, sessionId) =>
        registry.get(sessionId),
      );
    },

    /** Tear down every open tunnel — called on app quit. */
    dispose() {
      return registry.detachAll();
    },
  };
}
