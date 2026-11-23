/**
 * Whether the user is at this tab: it is visible and they touched it (a
 * pointer, a key, a scroll, the wheel) within {@link USER_IDLE_AFTER_MS}.
 * A tab left open and forgotten, visible or not, reads as idle, so
 * background refreshes can wait for the user to come back.
 */
export const USER_IDLE_AFTER_MS = 10 * 60_000;

const ACTIVITY_EVENTS = ["pointerdown", "keydown", "wheel", "scroll"] as const;

let lastActivity = Date.now();
let listeners = 0;
const onReturn = new Set<() => void>();

function noteActivity() {
  const wasIdle = isUserIdle();
  lastActivity = Date.now();
  if (wasIdle) onReturn.forEach((callback) => callback());
}

function onVisibility() {
  if (document.visibilityState !== "visible") return;
  lastActivity = Date.now();
  onReturn.forEach((callback) => callback());
}

export function isUserIdle(now = Date.now()): boolean {
  if (globalThis.document?.visibilityState === "hidden") return true;
  return now - lastActivity > USER_IDLE_AFTER_MS;
}

/**
 * Track the user's activity while anything wants it, and call `onBack` when
 * they come back: the tab is shown again, or they touch it after being idle.
 * Returns the function that stops it.
 */
export function watchUserActivity(onBack: () => void): () => void {
  onReturn.add(onBack);
  if (listeners === 0) {
    lastActivity = Date.now();
    ACTIVITY_EVENTS.forEach((type) =>
      window.addEventListener(type, noteActivity, {
        passive: true,
        capture: true,
      }),
    );
    document.addEventListener("visibilitychange", onVisibility);
  }
  listeners += 1;
  return () => {
    onReturn.delete(onBack);
    listeners -= 1;
    if (listeners === 0) {
      ACTIVITY_EVENTS.forEach((type) =>
        window.removeEventListener(type, noteActivity, { capture: true }),
      );
      document.removeEventListener("visibilitychange", onVisibility);
    }
  };
}
