/**
 * Navigation and popup policy for the Electron shell.
 *
 * `window-url-policy.mjs` answers "what kind of URL is this?". This module
 * answers "what should the shell do when a window tries to go there?", for
 * the three entry points a renderer can reach:
 *
 *   - top-level navigation of a window (`will-navigate` / `will-redirect`),
 *   - `window.open()` from the main window (`setWindowOpenHandler`),
 *   - the OAuth popup, which is same-origin `about:blank` and therefore has
 *     all three of the above available to it in turn.
 *
 * The rules live here rather than inline in main.mjs so they can be tested
 * without launching Electron: every function takes the `openExternal`
 * implementation as an argument, and the window/webContents objects are used
 * only through `on()`, `setWindowOpenHandler()` and `close()`.
 */

import {
  isExternalBrowsableUrl,
  isLoopbackAppUrl,
} from "./window-url-policy.mjs";

/**
 * Hand a URL to the OS browser / protocol handler, but only for the schemes
 * the policy allows. `shell.openExternal` forwards anything it is given, so
 * `file:`, `smb:` and custom scheme handlers must never reach it.
 */
export function openExternalUrl(url, openExternal) {
  if (isExternalBrowsableUrl(url)) {
    openExternal(url);
  }
}

/**
 * Keep a webContents pinned to the loopback app. Any other top-level
 * navigation (including one reached through a server-side redirect) is
 * cancelled and handed to the system browser instead.
 */
export function attachNavigationGuard(webContents, openExternal) {
  const guard = (event, url) => {
    if (isLoopbackAppUrl(url)) return;
    event.preventDefault();
    openExternalUrl(url, openExternal);
  };
  webContents.on("will-navigate", guard);
  // A loopback URL that 302s to a remote host never fires `will-navigate` for
  // the redirect target, so the guard has to cover redirects as well.
  webContents.on("will-redirect", guard);
}

/**
 * The `setWindowOpenHandler` policy for the main window.
 *
 * `about:blank` must be allowed: the "Login with OpenHands Cloud" device flow
 * opens it on the user's click (to beat popup blockers) and navigates it to
 * the verification URL once it has one. `attachPopupPolicy` below governs the
 * popup from that point on.
 */
export function mainWindowOpenHandler(url, openExternal) {
  if (url === "about:blank") {
    return {
      action: "allow",
      overrideBrowserWindowOptions: { width: 800, height: 700 },
    };
  }
  if (isLoopbackAppUrl(url)) {
    return { action: "allow" };
  }
  openExternalUrl(url, openExternal);
  return { action: "deny" };
}

/**
 * Apply the shell's policy to a popup created by the main window.
 *
 * The popup starts at `about:blank` and is same-origin with the app, so it
 * has three ways to reach remote content and each needs the same answer:
 * navigating itself, being redirected, and calling `window.open()` (which
 * inherits no handler from the parent and defaults to allowing).
 */
export function attachPopupPolicy(popupWin, openExternal) {
  const handOff = (event, url) => {
    if (url === "about:blank" || isLoopbackAppUrl(url)) return;
    event.preventDefault();
    openExternalUrl(url, openExternal);
    popupWin.close();
  };
  popupWin.webContents.on("will-navigate", handOff);
  popupWin.webContents.on("will-redirect", handOff);
  popupWin.webContents.setWindowOpenHandler(({ url }) => {
    if (isLoopbackAppUrl(url)) return { action: "allow" };
    openExternalUrl(url, openExternal);
    return { action: "deny" };
  });
}
