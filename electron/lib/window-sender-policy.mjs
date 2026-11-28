/**
 * Trust policy for MARS IPC senders: only the main app window's webContents
 * may call the MARS tunnel bridge methods. The loading window, the OAuth
 * popup, or any other renderer in the loopback origin are denied.
 *
 * Pure — no Electron imports — so it can be unit tested without Electron.
 */

/**
 * True only when `event?.sender` exists, the window from `getWindow()`
 * exists and is not destroyed, and `event.sender === window.webContents`.
 * `getWindow` is read at call time because `mainWin` is recreated after
 * `activate`.
 */
export function isTrustedWindowSender(event, getWindow) {
  if (!event?.sender) return false;
  const window = getWindow?.();
  if (!window || window.isDestroyed()) return false;
  return event.sender === window.webContents;
}
