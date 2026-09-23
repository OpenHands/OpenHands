/**
 * Preload for the main application window.
 *
 * Exposes the MARS tunnel bridge to the renderer while keeping
 * contextIsolation intact — page JavaScript never gets direct network access
 * to harness-api or the bearer token; every call is a request to the main
 * process, which holds the actual tunnel (see scripts/mars-tunnel-bridge.mjs).
 *
 * Channel names mirror MARS_TUNNEL_IPC in scripts/mars-tunnel-bridge.mjs.
 * They are duplicated as literals because a sandboxed preload is CommonJS and
 * cannot import that ESM module.
 *
 * CommonJS on purpose: sandboxed preload scripts cannot use ESM.
 */
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("marsBridge", {
  /**
   * Opens (or reuses) a port-forward tunnel for one session and resolves
   * once its local listener is up: `{sessionId, status, remotePort,
   * localPort, error}`.
   */
  openTunnel: (params) => ipcRenderer.invoke("mars:openTunnel", params),
  /** Closes the tunnel for one session, if any. */
  closeTunnel: (sessionId) => ipcRenderer.invoke("mars:closeTunnel", sessionId),
  /** Reads a session's current tunnel status without opening or closing it. */
  getTunnel: (sessionId) => ipcRenderer.invoke("mars:getTunnel", sessionId),
});
