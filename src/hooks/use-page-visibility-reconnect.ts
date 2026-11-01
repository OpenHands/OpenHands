import { useEffect, useRef } from "react";
import { isMobileUserAgent } from "#/utils/utils";

/**
 * Grace period between the tab going hidden and us tearing the socket down.
 * Absorbs a quick app-switcher glance or alt-tab without touching the
 * connection; a mobile browser that is actually about to freeze the page
 * gives JS at least this long to run first, so a genuine backgrounding still
 * gets caught well before the OS kills the socket itself.
 */
const HIDE_DEBOUNCE_MS = 3_000;

export interface UsePageVisibilityReconnectOptions {
  /** Skip wiring listeners entirely when there is nothing to manage yet. */
  enabled: boolean;
  /**
   * Close the socket(s) cleanly. The debounced `visibilitychange`-hidden path
   * only calls this on mobile user agents: desktop browsers keep sockets
   * alive across ordinary tab switches, so tearing them down there would stop
   * live updates for no reason. The immediate `freeze` path calls this on any
   * platform, since by the time `freeze` fires the browser has already
   * decided to suspend the page. Either way, a clean (code 1000) close here
   * beats letting the OS discard the tab, which often surfaces as a non-1000
   * close and an error toast once the hook catches up.
   */
  disconnect: () => void;
  /**
   * Re-check connection health and reconnect anything that isn't open. Runs
   * on every return to the foreground, on every platform — a socket can die
   * in the background for reasons that have nothing to do with our own
   * teardown (OS sleep, a network blip, a proxy timeout), and this is the
   * single hook that catches all of them. Reconnecting bypasses the normal
   * backoff timer, which may be hours stale after the tab was frozen.
   */
  reconnectIfStale: () => void;
}

/**
 * Ties the conversation WebSocket's lifecycle to the page's visibility so a
 * mobile background/foreground cycle produces one clean disconnect and one
 * explicit, immediate reconnect — instead of the OS silently killing the
 * socket and an exponential-backoff timer (frozen along with everything else
 * while hidden) replaying a burst of stale retries once the timers unfreeze.
 */
export function usePageVisibilityReconnect({
  enabled,
  disconnect,
  reconnectIfStale,
}: UsePageVisibilityReconnectOptions): void {
  // Refs so the effect below doesn't need `disconnect`/`reconnectIfStale` in
  // its dependency array — both are recreated most renders, and re-running
  // the effect on every one of those would thrash the listeners for no
  // behavioral change.
  const disconnectRef = useRef(disconnect);
  disconnectRef.current = disconnect;
  const reconnectRef = useRef(reconnectIfStale);
  reconnectRef.current = reconnectIfStale;

  useEffect(() => {
    if (!enabled) return undefined;

    let hideTimeout: ReturnType<typeof setTimeout> | null = null;
    const clearHideTimeout = () => {
      if (hideTimeout !== null) {
        clearTimeout(hideTimeout);
        hideTimeout = null;
      }
    };

    const handleHidden = () => {
      if (!isMobileUserAgent()) return;
      // Restart rather than stack: rapid flapping keeps pushing this out
      // rather than ever accumulating a second pending teardown.
      clearHideTimeout();
      hideTimeout = setTimeout(() => {
        hideTimeout = null;
        disconnectRef.current();
      }, HIDE_DEBOUNCE_MS);
    };

    const handleVisible = () => {
      // A return inside the grace period cancels the pending teardown before
      // it ever runs — the flap guard: quick background/foreground cycles
      // never touch the socket at all.
      clearHideTimeout();
      reconnectRef.current();
    };

    const handleVisibilityChange = () => {
      if (document.visibilityState === "hidden") {
        handleHidden();
      } else {
        handleVisible();
      }
    };

    // `freeze` (Page Lifecycle API; Chromium only) fires synchronously right
    // before the browser suspends JS execution entirely, so there is no time
    // left to debounce — any pending timer from `handleHidden` would lose
    // this race, since a frozen page can't run its own `setTimeout`
    // callback. No user-agent check either: once the browser has decided to
    // freeze the page (immediately on mobile, or after a long-hidden desktop
    // tab), the socket is already on borrowed time regardless of platform.
    const handleFreeze = () => {
      clearHideTimeout();
      disconnectRef.current();
    };

    // `pagehide` fires when iOS Safari discards a backgrounded tab outright,
    // which is not guaranteed to be preceded by `visibilitychange`; `pageshow`
    // and `resume` are its restore counterparts, including a bfcache restore.
    document.addEventListener("visibilitychange", handleVisibilityChange);
    window.addEventListener("pagehide", handleHidden);
    window.addEventListener("pageshow", handleVisible);
    document.addEventListener("freeze", handleFreeze);
    document.addEventListener("resume", handleVisible);

    return () => {
      clearHideTimeout();
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      window.removeEventListener("pagehide", handleHidden);
      window.removeEventListener("pageshow", handleVisible);
      document.removeEventListener("freeze", handleFreeze);
      document.removeEventListener("resume", handleVisible);
    };
  }, [enabled]);
}
