/**
 * Pick which window to bring forward when a second Electron instance
 * launches. Prefers the main app window, falls back to the loading
 * splash, and returns null when neither exists.
 *
 * Pure — no Electron imports — so it can be unit tested without Electron.
 */

/**
 * @param {Electron.BrowserWindow | null | undefined} mainWin
 * @param {Electron.BrowserWindow | null | undefined} loadingWin
 * @returns {Electron.BrowserWindow | null}
 */
export function chooseWindowToFocus(mainWin, loadingWin) {
  if (mainWin && !mainWin.isDestroyed()) return mainWin;
  if (loadingWin && !loadingWin.isDestroyed()) return loadingWin;
  return null;
}
