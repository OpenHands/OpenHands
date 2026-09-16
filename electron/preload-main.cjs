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

contextBridge.exposeInMainWorld("desktopShell", {
  platform: process.platform,
  /**
   * Subscribe to native fullscreen transitions: cb(isFullScreen). The main
   * process also replays the current state on load. Returns an unsubscribe fn.
   *
   * Chromium does not report `display-mode: fullscreen` for a natively
   * fullscreened BrowserWindow, so a CSS media query cannot see this.
   */
  onFullScreenChange(cb) {
    if (typeof cb !== "function") return () => {};
    const listener = (_event, isFullScreen) => cb(Boolean(isFullScreen));
    ipcRenderer.on("window:full-screen", listener);
    return () => ipcRenderer.removeListener("window:full-screen", listener);
  },
});
