/**
 * Preload for the main application window.
 *
 * Exposes the MARS bridge to the renderer while keeping contextIsolation
 * intact — page JavaScript never gets direct network access to harness-api
 * or the bearer token; every call is a request to the main process, which
 * holds the credentials and the actual tunnels (see
 * scripts/mars-tunnel-bridge.mjs).
 *
 * Channel names mirror MARS_TUNNEL_IPC in scripts/mars-tunnel-bridge.mjs.
 * They are duplicated as literals because a sandboxed preload is CommonJS and
 * cannot import that ESM module.
 *
 * CommonJS on purpose: sandboxed preload scripts cannot use ESM.
 */
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("marsBridge", {
  /** Stored connections plus whether OAuth is configured in this build. */
  getAuthState: () => ipcRenderer.invoke("mars:getAuthState"),
  /** Opens the system browser; resolves once the grant comes back. */
  signInWithOAuth: () => ipcRenderer.invoke("mars:signInWithOAuth"),
  savePat: (payload) => ipcRenderer.invoke("mars:savePat", payload),
  setActiveConnection: (id) =>
    ipcRenderer.invoke("mars:setActiveConnection", id),
  signOut: (id) => ipcRenderer.invoke("mars:signOut", id),

  listSessions: (options) => ipcRenderer.invoke("mars:listSessions", options),
  /** Durable agent definitions; one config has many sessions. */
  listAgentConfigs: (options) =>
    ipcRenderer.invoke("mars:listAgentConfigs", options),
  listConfigSessions: (configId, options) =>
    ipcRenderer.invoke("mars:listConfigSessions", configId, options),
  /** Resolves once the new session is READY. */
  createSession: (configId, name) =>
    ipcRenderer.invoke("mars:createSession", configId, name),
  pauseSession: (sessionId) =>
    ipcRenderer.invoke("mars:pauseSession", sessionId),
  resumeSession: (sessionId) =>
    ipcRenderer.invoke("mars:resumeSession", sessionId),

  /**
   * Opens (or reuses) the tunnel to a session's agent-server and resolves once
   * its local listener is up: `{sessionId, status, remotePort, localPort,
   * error}`.
   */
  openTunnel: (params) => ipcRenderer.invoke("mars:openTunnel", params),
  /** Closes the tunnel for one session, if any. */
  closeTunnel: (sessionId) => ipcRenderer.invoke("mars:closeTunnel", sessionId),
  /** Reads a session's current tunnel status without opening or closing it. */
  getTunnel: (sessionId) => ipcRenderer.invoke("mars:getTunnel", sessionId),
});
