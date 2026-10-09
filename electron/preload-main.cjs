/**
 * Preload for the main app window.
 *
 * The renderer is the ordinary web app served over loopback, so it cannot tell
 * a desktop launch from a browser tab. This exposes just enough for the shell
 * to reserve room for the macOS traffic lights and mark a drag region — see
 * src/utils/desktop-shell.ts.
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
