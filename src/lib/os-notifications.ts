/**
 * Thin wrapper around the browser `Notification` API and the experimental
 * `navigator.setAppBadge` / `clearAppBadge` APIs. Degrades gracefully when
 * either is unavailable (SSR, insecure context, unsupported browser, denied
 * permission). Every public entry point is safe to call unconditionally from
 * component code.
 *
 * A single named owner: this module. Components never touch `Notification`
 * or `navigator.setAppBadge` directly.
 */

export type NotificationPermissionState =
  | "unsupported"
  | "default"
  | "granted"
  | "denied";

export interface AttentionNotificationPayload {
  conversationId: string;
  /** Conversation title shown in the notification body. */
  conversationTitle: string;
  /** Short outcome label, e.g. "Finished", "Needs input", "Error". */
  outcome: string;
  /** Optional longer description for the notification body. */
  description?: string;
  /** Router path to navigate to when the notification is clicked. */
  targetPath?: string;
}

/** True when the runtime exposes a browser `Notification` constructor. */
export function isNotificationApiAvailable(): boolean {
  return typeof window !== "undefined" && typeof Notification !== "undefined";
}

/** True when `navigator.setAppBadge` / `clearAppBadge` are callable. */
export function isBadgeApiAvailable(): boolean {
  return (
    typeof navigator !== "undefined" &&
    typeof navigator.setAppBadge === "function" &&
    typeof navigator.clearAppBadge === "function"
  );
}

/** True when the document is currently hidden or blurred. */
export function isDocumentInBackground(): boolean {
  if (typeof document === "undefined") return false;
  if (document.visibilityState === "hidden") return true;
  // `document.hasFocus()` returns false when the window itself isn't focused,
  // even if the tab is visible (e.g. user switched to another app).
  return typeof document.hasFocus === "function" && !document.hasFocus();
}

/**
 * One-shot request for OS notification permission. MUST be invoked in
 * response to an explicit user gesture (click on the settings toggle).
 * Returns the resulting permission state — never throws when the API is
 * missing.
 */
export async function requestNotificationPermission(): Promise<NotificationPermissionState> {
  if (!isNotificationApiAvailable()) return "unsupported";
  try {
    const result = await Notification.requestPermission();
    return (result ?? Notification.permission) as NotificationPermissionState;
  } catch {
    // Safari <= 15 may throw when `requestPermission` is called from a
    // non-page context; fall back to the static permission string.
    return (Notification.permission ??
      "default") as NotificationPermissionState;
  }
}

/** Current permission state without prompting. */
export function getNotificationPermission(): NotificationPermissionState {
  if (!isNotificationApiAvailable()) return "unsupported";
  return (Notification.permission ?? "default") as NotificationPermissionState;
}

/**
 * Fire one OS notification. `tag` is derived from `conversationId` so
 * repeated attention transitions for the same conversation replace the
 * existing notification rather than stacking. Returns the created
 * `Notification` object (useful in tests) or `null` if notifications are
 * unavailable / not permitted.
 */
export function fireAttentionNotification(
  payload: AttentionNotificationPayload,
): Notification | null {
  if (getNotificationPermission() !== "granted") return null;
  if (!isNotificationApiAvailable()) return null;

  const title = `${payload.conversationTitle || "Agent"} — ${payload.outcome}`;
  const body =
    payload.description ??
    (payload.conversationTitle
      ? `The conversation "${payload.conversationTitle}" ${payload.outcome.toLowerCase()}.`
      : `The agent ${payload.outcome.toLowerCase()}.`);

  let notification: Notification;
  try {
    notification = new Notification(title, {
      body,
      tag: `openhands-attention-${payload.conversationId}`,
      silent: false,
    });
  } catch {
    return null;
  }

  if (payload.targetPath) {
    notification.onclick = () => {
      try {
        window.focus();
        if ("open" in notification) {
          // Some browsers expose notification.open — useful in service workers.
          try {
            (notification as unknown as { open?: () => void }).open?.();
          } catch {
            // ignore
          }
        }
      } catch {
        // ignore focus errors
      }
    };
  }

  return notification;
}

/**
 * Set the app badge to `count` awaiting-attention conversations. A value of
 * `0` clears the badge. No-op when the Badge API is absent.
 */
export function setAttentionBadge(count: number): void {
  if (!isBadgeApiAvailable()) return;
  try {
    if (count <= 0) {
      navigator.clearAppBadge?.();
    } else {
      navigator.setAppBadge!(count);
    }
  } catch {
    // Badge API may reject with a SecurityError (not installed PWA, etc.).
    // Swallow — this is cosmetic, never fatal.
  }
}
