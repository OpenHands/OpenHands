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
   * The window's current fullscreen state. A subscriber starts listening long
   * after the window opened, so it can miss the transition it is already past:
   * a reload of a window that is fullscreen reports no event at all.
   */
  getFullScreen: () => ipcRenderer.invoke("window:full-screen:get"),
  /**
   * Subscribe to native fullscreen transitions: cb(isFullScreen). Returns an
   * unsubscribe fn.
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
