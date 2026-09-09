/**
 * Runtime detection for the packaged Electron shell.
 *
 * The main window has no preload, so user-agent is the only signal. macOS
 * uses `titleBarStyle: hiddenInset`, which overlays traffic lights on the
 * renderer — the frontend must reserve a drag strip for them. Windows/Linux
 * keep a native title bar, so they do not need that strip.
 */
export function isElectronShell(): boolean {
  return (
    typeof navigator !== "undefined" && /Electron/i.test(navigator.userAgent)
  );
}

export function needsDesktopTitlebar(): boolean {
  return (
    isElectronShell() &&
    typeof navigator !== "undefined" &&
    /Macintosh|Mac OS X/i.test(navigator.userAgent)
  );
}
