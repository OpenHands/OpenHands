// Module-level store for the environment-switch overlay.
//
// This file is intentionally separate from `environment-switch-overlay.tsx`
// (which contains the React component and an inline SVG icon) so that the
// always-mounted sidebar can import `triggerEnvironmentSwitch` /
// `dismissEnvironmentSwitch` without dragging the overlay's render code
// into the eager graph. The overlay component is loaded lazily from
// `routes/root-layout.tsx`.
//
// The overlay state must survive the unmount of the component that triggers a
// switch — the user-context menu remounts (`menuResetCount` key flip in
// user-actions.tsx) the moment the dropdown's portaled option list is
// clicked, because that click registers as outside the menu's
// `useClickOutsideElement` ref. If state lived inside BackendSelector, the
// trigger would fire and immediately get torn down before React paints the
// overlay.

export const ENVIRONMENT_SWITCH_DURATION_MS = 980;
export const ENVIRONMENT_SWITCH_SETACTIVE_DELAY_MS = 400;

export interface EnvironmentSwitchSnapshot {
  visible: boolean;
  target: string;
}

let snapshot: EnvironmentSwitchSnapshot = { visible: false, target: "" };
const listeners = new Set<() => void>();
let hideTimeoutId: ReturnType<typeof setTimeout> | null = null;

function setSnapshot(next: EnvironmentSwitchSnapshot) {
  snapshot = next;
  if (typeof document !== "undefined") {
    if (next.visible) {
      document.body.setAttribute("data-environment-switching", "true");
    } else {
      document.body.removeAttribute("data-environment-switching");
    }
  }
  listeners.forEach((listener) => listener());
}

export function triggerEnvironmentSwitch(target: string) {
  setSnapshot({ visible: true, target });
  if (hideTimeoutId) clearTimeout(hideTimeoutId);
  hideTimeoutId = setTimeout(() => {
    setSnapshot({ visible: false, target: "" });
    hideTimeoutId = null;
  }, ENVIRONMENT_SWITCH_DURATION_MS);
}

export function dismissEnvironmentSwitch() {
  if (hideTimeoutId) {
    clearTimeout(hideTimeoutId);
    hideTimeoutId = null;
  }
  setSnapshot({ visible: false, target: "" });
}

export function subscribeEnvironmentSwitch(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getEnvironmentSwitchSnapshot() {
  return snapshot;
}

/** Test-only: clear the pending hide timer and reset the snapshot. */

export function __resetEnvironmentSwitchOverlayForTests() {
  if (hideTimeoutId) {
    clearTimeout(hideTimeoutId);
    hideTimeoutId = null;
  }
  setSnapshot({ visible: false, target: "" });
}
