/**
 * Preload for the loading window (loading.html).
 *
 * Bridges the startup-log console to the main process over IPC while keeping
 * contextIsolation (and the default renderer sandbox) intact. The one-line
 * status headline intentionally does NOT go through here — main.mjs sets it
 * via executeJavaScript → window.__setLoadingStatus (see setLoadingStatus).
 *
 * CommonJS on purpose: sandboxed preload scripts cannot use ESM.
 */
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("desktopBoot", {
  /** Subscribe to batched startup-log lines: cb([{name, line, level}, …]). */
  onLogBatch(cb) {
    if (typeof cb !== "function") return;
    ipcRenderer.on("boot-log:batch", (_event, batch) => cb(batch));
  },
  /** Subscribe to the fatal startup-failure notification: cb(summary). */
  onFatal(cb) {
    if (typeof cb !== "function") return;
    ipcRenderer.on("boot-log:fatal", (_event, summary) => cb(summary));
  },
  /** Grow/shrink the window to reveal or hide the console panel. */
  setDetailsExpanded: (expanded) =>
    ipcRenderer.invoke("boot-log:set-expanded", Boolean(expanded)),
  /** Copy the full buffered startup log to the clipboard. */
  copyLogs: () => ipcRenderer.invoke("boot-log:copy"),
  /** Quit the app (failure-state action; the frameless splash has no close UI). */
  quit: () => ipcRenderer.invoke("boot-log:quit"),
});
