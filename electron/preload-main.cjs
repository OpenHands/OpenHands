/**
 * Preload for the main app window.
 *
 * The renderer is the ordinary web app served over loopback, so it cannot tell
 * a desktop launch from a browser tab. This exposes just enough for the shell
 * to reserve room for the macOS traffic lights and mark a drag region — see
 * src/utils/desktop-shell.ts.
 *
 * It also exposes the MARS bridge (`window.marsBridge`). Page JavaScript never
 * gets direct network access to harness-api or the bearer token; every call is
 * a request to the main process, which holds the credentials and the tunnels
 * (see scripts/mars-tunnel-bridge.mjs). Its channel names mirror
 * MARS_TUNNEL_IPC there, duplicated as literals because this preload cannot
 * import that ESM module.
 *
 * The Maritime bridge (`window.maritimeBridge`) works the same way: the
 * Maritime API key stays in the main process (scripts/maritime-bridge.mjs),
 * and its channel names mirror MARITIME_IPC there.
 *
 * CommonJS on purpose: sandboxed preload scripts cannot use ESM.
 */
const { contextBridge, ipcRenderer } = require("electron");

// Read once here, before any page script runs, and kept current from the
// pushed transitions: the renderer subscribes long after load, and a reload of
// a fullscreen window reports no transition at all.
let isFullScreen = Boolean(ipcRenderer.sendSync("window:full-screen:get"));
const listeners = new Set();
ipcRenderer.on("window:full-screen", (_event, value) => {
  isFullScreen = Boolean(value);
  for (const cb of listeners) cb(isFullScreen);
});

contextBridge.exposeInMainWorld("desktopShell", {
  platform: process.platform,
  /** The window's current fullscreen state. */
  isFullScreen: () => isFullScreen,
  /**
   * Subscribe to native fullscreen transitions: cb(isFullScreen). Returns an
   * unsubscribe fn.
   *
   * Chromium does not report `display-mode: fullscreen` for a natively
   * fullscreened BrowserWindow, so a CSS media query cannot see this.
   */
  onFullScreenChange(cb) {
    listeners.add(cb);
    return () => listeners.delete(cb);
  },
});

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
  /** Sandbox sizes a new agent may pick, smallest first. */
  listSandboxSizes: () => ipcRenderer.invoke("mars:listSandboxSizes"),
  /** `{name, size?}`; the main process owns the OpenHands manifest. */
  createOpenHandsAgent: (payload) =>
    ipcRenderer.invoke("mars:createOpenHandsAgent", payload),
  /** Resolves once the new session is READY. */
  createSession: (configId, name) =>
    ipcRenderer.invoke("mars:createSession", configId, name),
  pauseSession: (sessionId) =>
    ipcRenderer.invoke("mars:pauseSession", sessionId),
  resumeSession: (sessionId) =>
    ipcRenderer.invoke("mars:resumeSession", sessionId),
  /** Ends the session and its sandbox for good; drops any live connection first. */
  destroySession: (sessionId) =>
    ipcRenderer.invoke("mars:destroySession", sessionId),
  /** Soft-deletes the agent; the renderer only offers it when no session is live. */
  deleteAgentConfig: (configId) =>
    ipcRenderer.invoke("mars:deleteAgentConfig", configId),

  /**
   * Connects to a session's agent-server over a port-forward tunnel and
   * resolves once it is reachable: `{sessionId, status, transport, host, remotePort,
   * localPort, error}`. `host` is the base URL to register as the backend.
   */
  openTunnel: (params) => ipcRenderer.invoke("mars:openTunnel", params),
  /** Forgets the connection for one session (and closes its tunnel, if any). */
  closeTunnel: (sessionId) => ipcRenderer.invoke("mars:closeTunnel", sessionId),
  /** Reads a session's current connection status without changing it. */
  getTunnel: (sessionId) => ipcRenderer.invoke("mars:getTunnel", sessionId),
});

contextBridge.exposeInMainWorld("maritimeBridge", {
  /** `{isSignedIn, isSaved, storage}`; never the key itself. */
  getAuthState: () => ipcRenderer.invoke("maritime:getAuthState"),
  /** `{apiKey}`; kept only once Maritime accepts it. */
  saveApiKey: (payload) => ipcRenderer.invoke("maritime:saveApiKey", payload),
  signOut: () => ipcRenderer.invoke("maritime:signOut"),
  /** Plan, machine counts and the memory range a new agent may use. */
  getAccount: () => ipcRenderer.invoke("maritime:getAccount"),
  /** The OpenHands agents Canvas created on this Maritime account. */
  listAgents: () => ipcRenderer.invoke("maritime:listAgents"),
  /** `{name, memMb, origin}`; the main process owns the rest of the request. */
  createAgent: (payload) => ipcRenderer.invoke("maritime:createAgent", payload),
  /** Wakes the agent; resolves `{agentId, name, host, sessionApiKey}`. */
  openAgent: (agentId) => ipcRenderer.invoke("maritime:openAgent", agentId),
  /** Sets a new session key and restarts the agent-server. */
  resetAccessKey: (agentId) =>
    ipcRenderer.invoke("maritime:resetAccessKey", agentId),
  sleepAgent: (agentId) => ipcRenderer.invoke("maritime:sleepAgent", agentId),
  /** Deletes the agent and its disk on Maritime. */
  deleteAgent: (agentId) => ipcRenderer.invoke("maritime:deleteAgent", agentId),
});
