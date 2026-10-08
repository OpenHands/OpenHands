/**
 * Desktop-shell detection for the renderer.
 *
 * The app is served over loopback, so a desktop launch is indistinguishable
 * from a browser tab unless the Electron preload says so — see
 * electron/preload-main.cjs.
 */

interface DesktopShellBridge {
  platform?: string;
  isFullScreen?: () => boolean;
  onFullScreenChange?: (cb: (isFullScreen: boolean) => void) => () => void;
}

declare global {
  interface Window {
    desktopShell?: DesktopShellBridge;
  }
}

function getDesktopShell(): DesktopShellBridge | undefined {
  return typeof window === "undefined" ? undefined : window.desktopShell;
}

/**
 * True inside the macOS desktop app, where `titleBarStyle: "hiddenInset"`
 * leaves the traffic lights floating over the top-left of the app shell.
 */
export function isMacDesktopShell(): boolean {
  return getDesktopShell()?.platform === "darwin";
}

/** Whether the desktop window is in native fullscreen; false in a browser tab. */
export function isDesktopFullScreen(): boolean {
  return getDesktopShell()?.isFullScreen?.() ?? false;
}

/**
 * Subscribe to native fullscreen transitions. Returns an unsubscribe function;
 * outside the desktop app it is a no-op, since a browser tab owns no window.
 */
export function subscribeDesktopFullScreen(
  cb: (isFullScreen: boolean) => void,
): () => void {
  return getDesktopShell()?.onFullScreenChange?.(cb) ?? (() => {});
}
