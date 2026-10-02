/**
 * Desktop-shell detection for the renderer.
 *
 * The app is served over loopback, so a desktop launch is indistinguishable
 * from a browser tab unless the Electron preload says so — see
 * electron/preload-main.cjs.
 */

interface DesktopShellBridge {
  platform?: string;
  getFullScreen?: () => Promise<boolean>;
  onFullScreenChange?: (cb: (isFullScreen: boolean) => void) => () => void;
}

function getDesktopShell(): DesktopShellBridge | undefined {
  if (typeof window === "undefined") return undefined;
  return (window as Window & { desktopShell?: DesktopShellBridge })
    .desktopShell;
}

/**
 * True inside the macOS desktop app, where `titleBarStyle: "hiddenInset"`
 * leaves the traffic lights floating over the top-left of the app shell.
 */
export function isMacDesktopShell(): boolean {
  return getDesktopShell()?.platform === "darwin";
}

/**
 * Subscribe to native fullscreen transitions and report the current state once.
 * Returns an unsubscribe function; outside the desktop app it is a no-op, since
 * a browser tab owns no window.
 *
 * The current state has to be read rather than pushed: this runs after
 * hydration, well past the point where the window could have announced a
 * fullscreen it entered before the page loaded.
 */
export function subscribeDesktopFullScreen(
  cb: (isFullScreen: boolean) => void,
): () => void {
  const shell = getDesktopShell();
  if (!shell) return () => {};

  let initialReadIsCurrent = true;
  const unsubscribe = shell.onFullScreenChange?.((isFullScreen) => {
    initialReadIsCurrent = false;
    cb(isFullScreen);
  });

  shell
    .getFullScreen?.()
    .then((isFullScreen) => {
      if (initialReadIsCurrent) cb(Boolean(isFullScreen));
    })
    .catch(() => {});

  return () => {
    initialReadIsCurrent = false;
    unsubscribe?.();
  };
}
